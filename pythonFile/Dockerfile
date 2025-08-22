# Dùng base image chính thức của Python 3.12, bản slim (nhẹ, không có nhiều gói hệ thống cài sẵn). Từ đây, mọi thứ bạn cài đặt sẽ dựa trên image này.
FROM python:3.12-slim

# Cài đặt thư viện hệ thống
# apt-get update cập nhật danh sách package.
# Cài libgl1 và libglib2.0-0 → bắt buộc để OpenCV có thể đọc ảnh/hiển thị (nếu không sẽ lỗi khi cv2.imread hoặc cv2.VideoCapture).
# rm -rf /var/lib/apt/lists/* để xoá cache → giúp image nhẹ hơn.
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Cài dependencies, --no-cache-dir giúp không lưu cache → image nhẹ hơn.
RUN pip install --no-cache-dir torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1 --extra-index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Đặt thư mục làm việc mặc định trong container là /TEST. Mọi lệnh tiếp theo (copy, run, cmd) sẽ thực hiện trong /TEST.
WORKDIR /test

# Copy toàn bộ project (từ thư mục hiện tại trên host) vào /TEST trong container. Nghĩa là main.py, mySQL_db.py, best.pt, v.v. đều sẽ nằm trong /TEST.
COPY . /test

# Thông báo rằng container sẽ lắng nghe ở cổng 8000 (FastAPI). Dùng chung với docker-compose.yml → "8000:8000" để mở ra ngoài.
EXPOSE 8000

# Khi container chạy, nó sẽ thực hiện lệnh này: python main.py. Tức là khởi động luôn FastAPI app của bạn.
CMD ["python", "main.py"]