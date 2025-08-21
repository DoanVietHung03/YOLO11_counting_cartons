import mysql.connector
from datetime import datetime
import json
import config

# Connect to server
db_config = {
    'host': "db",
    'port': 3306,
    'user': "root",
    'password': "12345",
}

DB_NAME = "rtsp_db"

CNX = mysql.connector.connect(**db_config)
CUR = CNX.cursor()

# Create database if not exist
CUR.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")

# Select database to work
CUR.execute(f"USE {DB_NAME}")

# Create table if it does not exist
CUR.execute("""CREATE TABLE IF NOT EXISTS detections (id INT AUTO_INCREMENT PRIMARY KEY,
                                                    timestamp DATETIME,
                                                    track_id INT,
                                                    label VARCHAR(255),
                                                    confidence FLOAT,
                                                    bbox_x1 INT,
                                                    bbox_y1 INT,
                                                    bbox_x2 INT,
                                                    bbox_y2 INT)""")
                                                    
def db_insert(detections_list):
    print(f"Trạng thái web khi ghi DB: {config.WEB_STATUS}")
    if config.WEB_STATUS:
        try:
            sql = """
            INSERT INTO detections
            (timestamp, track_id, label, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """

            for det in detections_list:
                CUR.execute(sql, (
                    det["timestamp"],
                    det["track_id"],
                    det["label"],
                    det["confidence"],
                    det["bbox"][0], det["bbox"][1], det["bbox"][2], det["bbox"][3]
                ))

            print(f"✅ Inserted: {det}")

            CNX.commit()

        except mysql.connector.Error as err:
            print(f"Error connecting to MySQL: {err}")
    else:
        print("Web is not connected, skipping DB insert and closing connection.")
        CUR.close()
        CNX.close()