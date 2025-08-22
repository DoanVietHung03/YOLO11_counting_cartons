import mysql.connector
from mysql.connector.pooling import MySQLConnectionPool
import logging

# Thiết lập logging
logger = logging.getLogger('database')

# Giữ thông tin cấu hình DB
DB_CONFIG = {
    'host': "db",
    'port': 3306,
    'user': "root",
    'password': "12345",
    'database': "rtsp_db"
}

# Khởi tạo connection pool
POOL_CONFIG = {
    "pool_name": "mypool",
    "pool_size": 5,  # Đủ cho n luồng RTSP
    **DB_CONFIG
}

try:
    db_pool = MySQLConnectionPool(**POOL_CONFIG)
    logger.info("Database connection pool initialized.")
except mysql.connector.Error as err:
    logger.error(f"Error initializing database connection pool: {err}")

def initialize_database(stream_ids: list):
    """
    Hàm này chạy một lần để đảm bảo DB và các bảng tồn tại.
    Tạo một bảng riêng cho mỗi stream_id trong danh sách.
    """
    try:
        # Kết nối không chỉ định database để tạo database
        temp_config = DB_CONFIG.copy()
        temp_config.pop('database', None)
        
        with mysql.connector.connect(**temp_config) as cnx:
            with cnx.cursor() as cur:
                cur.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
                logger.info(f"Database '{DB_CONFIG['database']}' ensured to exist.")
                
        # Kết nối vào database để tạo bảng cho mỗi stream
        with db_pool.get_connection() as cnx:
            with cnx.cursor() as cur:
                for stream_id in stream_ids:
                    # Tạo tên bảng động, thay thế ký tự không hợp lệ
                    table_name = f"detections_stream_{stream_id}"
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS {table_name} (
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            timestamp DATETIME,
                            track_id INT,
                            label VARCHAR(255),
                            confidence FLOAT
                        )
                    """)
                    logger.info(f"Table '{table_name}' ensured to exist.")

                    # Kiểm tra xem chỉ mục đã tồn tại chưa
                    cur.execute(f"""
                        SELECT COUNT(*) 
                        FROM information_schema.statistics 
                        WHERE table_name = '{table_name}' 
                        AND index_name = 'idx_timestamp'
                    """)
                    if cur.fetchone()[0] == 0:  # Chỉ mục chưa tồn tại
                        cur.execute(f"CREATE INDEX idx_timestamp ON {table_name} (timestamp)")
                        logger.info(f"Index idx_timestamp ensured for table '{table_name}'.")

            cnx.commit()
            logger.info("Database initialized successfully.")
    except mysql.connector.Error as err:
        logger.error(f"Failed to initialize database: {err}")
        raise

def db_insert(detections_list: list, stream_id):
    """
    Hàm insert một loạt detections vào bảng tương ứng với stream_id.
    Sử dụng connection pool để quản lý kết nối hiệu quả.
    Thread-safe và tối ưu cho đa luồng.
    """
    if not detections_list:
        return

    # Tạo tên bảng động
    table_name = f"detections_stream_{stream_id}"
    
    # Chuyển đổi dữ liệu sang dạng tuple để dùng executemany
    sql = f"INSERT INTO {table_name} (timestamp, track_id, label, confidence) VALUES (%s, %s, %s, %s)"
    data_to_insert = [
        (
            det["timestamp"],
            det["track_id"],
            det["label"],
            det["confidence"]
        )
        for det in detections_list
    ]
    
    cnx = None
    try:
        cnx = db_pool.get_connection()
        cur = cnx.cursor()
        
        # Dùng executemany để insert tất cả các dòng trong 1 lần
        cur.executemany(sql, data_to_insert)
        cnx.commit()
        logger.info(f"✅ Inserted {cur.rowcount} records into table '{table_name}'.")
        
    except mysql.connector.Error as err:
        logger.error(f"Database insert error for table '{table_name}': {err}")
        if cnx:
            cnx.rollback()
    finally:
        if cnx and cnx.is_connected():
            cur.close()
            cnx.close()