# 🧮 Math Instruct Llama

Dự án triển khai mô hình AI chuyên giải toán học, được Fine-tune (tinh chỉnh) từ mô hình gốc `unsloth/Llama-3.2-1B-Instruct`. Dự án hỗ trợ huấn luyện linh hoạt với nhiều phương pháp PEFT (QLoRA, LoRA, TinyLoRA) và được đóng gói hoàn chỉnh bằng **Docker** + **FastAPI** để sẵn sàng đưa vào môi trường thực tế (Production).

---

## 🐳 1. Hướng dẫn sử dụng bằng Docker (Nhanh nhất)

Nếu bạn chỉ muốn **chạy thử mô hình và gọi API** mà không cần cài đặt code, thư viện hay Python, hãy sử dụng Docker Image đã được đóng gói sẵn.

**Yêu cầu:** Máy đã cài đặt [Docker](https://docs.docker.com/get-docker/).

Mở Terminal và chạy dòng lệnh sau:
'''
sudo docker run --rm \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  nhannguyen2201/math-llama-app:v1
'''

💻 2. Hướng dẫn Dành cho Developer (Cài đặt & Huấn luyện)

git clone 

