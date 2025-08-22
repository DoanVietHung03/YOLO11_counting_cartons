import mysql.connector
import logging

# Thiết lập logging
logger = logging.getLogger('database')

# Giữ thông tin cấu hình DB ở đây, tách biệt khỏi logic
DB_CONFIG = {
    'host': "db",
    'port': 3306,
    'user': "root",
    'password': "12345",
    'database': "rtsp_db"
}

def initialize_database():
    """
    Hàm này chỉ chạy một lần để đảm bảo DB và bảng tồn tại.
    """
    try:
        # Kết nối không chỉ định database để tạo database
        temp_config = DB_CONFIG.copy()
        temp_config.pop('database', None)
        
        with mysql.connector.connect(**temp_config) as cnx:
            with cnx.cursor() as cur:
                cur.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
                logger.info(f"Database '{DB_CONFIG['database']}' ensured to exist.")
                
        # Kết nối vào database để tạo bảng
        with mysql.connector.connect(**DB_CONFIG) as cnx:
            with cnx.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS detections (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        timestamp DATETIME,
                        track_id INT,
                        label VARCHAR(255),
                        confidence FLOAT
                    )
                """)
                # Lưu ý: Tôi đã bỏ các cột bbox để khớp với dữ liệu bạn gửi vào DB
                # Nếu bạn cần bbox, hãy thêm lại các cột và điều chỉnh hàm db_insert
                logger.info("Table 'detections' ensured to exist.")

    except mysql.connector.Error as err:
        logger.error(f"Failed to initialize database: {err}")
        # Thoát nếu không thể khởi tạo DB
        raise

def db_insert(detections_list: list):
    """
    Hàm này thực hiện việc insert một loạt các detections vào database.
    Nó tự quản lý kết nối -> an toàn cho việc chạy trong các luồng riêng biệt.
    """
    # Không làm gì nếu danh sách rỗng
    if not detections_list:
        return

    # Tối ưu quan trọng: Chuyển đổi dữ liệu sang dạng tuple để dùng executemany
    sql = "INSERT INTO detections (timestamp, track_id, label, confidence) VALUES (%s, %s, %s, %s)"
    
    # Lưu ý: Dữ liệu bạn gửi vào trong code trước không có bbox.
    # Nếu object 'det' của bạn có key 'bbox', hãy sửa dòng dưới đây
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
        # Tối ưu: Tạo kết nối mới cho mỗi lần ghi batch -> Tránh lỗi timeout/stale connection
        cnx = mysql.connector.connect(**DB_CONFIG)
        cur = cnx.cursor()
        
        # Tối ưu: Dùng executemany để insert tất cả các dòng trong 1 lần -> Nhanh hơn nhiều
        cur.executemany(sql, data_to_insert)
        
        cnx.commit()
        logger.info(f"✅ Inserted {cur.rowcount} records into the database.")
        
    except mysql.connector.Error as err:
        logger.error(f"Database insert error: {err}")
        if cnx:
            cnx.rollback() # Hoàn tác nếu có lỗi
    finally:
        # Luôn đóng kết nối sau khi hoàn thành
        if cnx and cnx.is_connected():
            cur.close()
            cnx.close()