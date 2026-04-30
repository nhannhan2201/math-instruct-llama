# Dùng image Python 3.10 slim cho nhẹ, giống với môi trường Ubuntu của Nhan
FROM python:3.10-slim

# Thiết lập thư mục làm việc mặc định trong Container
WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết cho C++ compiler (dùng cho bnb, peft, v.v.)
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

# Copy file requirements vào trước để tận dụng Docker Cache
COPY requirements.txt .

# Cài đặt thư viện (thêm fastapi và uvicorn nếu trong requirements chưa có)
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn pydantic

# Bê toàn bộ Source code và thư mục Model vào container
# (Docker sẽ lấy thư mục models/qlora_final mà Nhan đã train)
COPY src/ src/
COPY models/ models/
COPY app.py .

# Mở port 8000 để giao tiếp với bên ngoài
EXPOSE 8000

# Lệnh khởi động server khi chạy Container
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]