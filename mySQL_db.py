import mysql.connector
from datetime import datetime
import json

def db_insert(detections_list):
    try:
        # Connect to server
        db_config = {
            'host': "localhost",
            'port': 3306,
            'user': "root",
            'password': "12345",
        }

        DB_NAME = "rtsp_db"

        cnx = mysql.connector.connect(**db_config)
        cur = cnx.cursor()

        # Create database if not exist
        cur.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")

        # Select database to work
        cnx.database = DB_NAME

        # Create table if it does not exist
        cur.execute("""CREATE TABLE IF NOT EXISTS detections (id INT AUTO_INCREMENT PRIMARY KEY,
                                                            timestamp DATETIME,
                                                            track_id INT,
                                                            label VARCHAR(255),
                                                            confidence FLOAT,
                                                            bbox_x1 INT,
                                                            bbox_y1 INT,
                                                            bbox_x2 INT,
                                                            bbox_y2 INT)""")

        sql = """
        INSERT INTO detections
        (timestamp, track_id, label, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """

        for det in detections_list:
            cur.execute(sql, (
                det["timestamp"],
                det["track_id"],
                det["label"],
                det["confidence"],
                det["bbox"][0], det["bbox"][1], det["bbox"][2], det["bbox"][3]
            ))

        cnx.commit()

        # # Fetch all detection records
        # cur.execute("SELECT * FROM detections")
        # detection_records = cur.fetchall()
        # print("Detection records: ", detection_records)

        cur.close()
        cnx.close()
        # print(f"✅ Insert {len(detections_list)} record(s) vào MySQL thành công")

    except mysql.connector.Error as err:
        print(f"❌ Error connecting to MySQL: {err}")