# Math Studio

Chatbot học toán sử dụng **Llama 3.2 1B Instruct**, fine-tune trên **MathInstruct** bằng **LoRA** và **QLoRA**. Web UI chạy trên máy local, kết nối tới model trên GPU Kaggle qua FastAPI và ngrok.

Giao diện hỗ trợ chọn model, hiển thị lời giải và công thức toán, sao chép đáp án và lưu lịch sử hội thoại trên trình duyệt.

## Giao diện

### LoRA · Tiếng Việt

![Math Studio — LoRA giải bài toán tiếng Việt](assets/mathstudio-lora-vietnamese.png)

### QLoRA · Tiếng Anh

![Math Studio — QLoRA giải bài toán tiếng Anh](assets/mathstudio-qlora-english.png)

## Chạy model trên GPU Kaggle

Các notebook nằm trong [notebooks/](notebooks/). Chọn notebook theo mục đích:

| Notebook | Dùng để làm gì? |
|---|---|
| [web-ui-peft.ipynb](notebooks/web-ui-peft.ipynb) | Khởi động hai server LoRA/QLoRA, gateway và ngrok để web UI gọi model |
| [inference-lora-qlora.ipynb](notebooks/inference-lora-qlora.ipynb) | Tải từng adapter và thử các bài toán trực tiếp trong notebook, không cần web UI |
| [full-training.ipynb](notebooks/full-training.ipynb) | Fine-tune LoRA/QLoRA, chạy validation và lưu adapter cùng MLflow tracking |

**Muốn chạy chatbot, dùng `web-ui-peft.ipynb`. Không cần huấn luyện lại nếu đã có adapter.**

1. Import notebook vào Kaggle, bật **Internet** và chọn **GPU T4 ×2**.
2. **Add Input** dataset chứa hai adapter đã train. Kiểm tra đường dẫn trong notebook khớp với Input của bạn; trọng số adapter không được lưu trong repo.
3. Lấy authtoken từ [ngrok](https://dashboard.ngrok.com/get-started/your-authtoken). Trong **Add-ons → Secrets** của Kaggle, lưu tên `NGROK_AUTHTOKEN` và cấp quyền cho notebook.
4. Chạy từng cell theo thứ tự: tải repo → kiểm tra GPU/adapter → cài thư viện → chạy QLoRA → chạy LoRA → chạy gateway → thử API → mở tunnel.
5. Sau mỗi lần khởi động, đọc log và chờ server sẵn sàng trước khi chạy tiếp. Hai request thử qua gateway cần trả HTTP `200`.
6. Lấy **URL gốc ngrok** được in ở cell cuối để kết nối web UI. Mở URL `/docs` để thử API từ trình duyệt.

Chỉ chạy mỗi cell khởi động server hoặc tunnel một lần trong một session để tránh trùng port. **Giữ session Kaggle hoạt động khi dùng web**; lưu notebook không giữ server chạy liên tục. Token phải giữ riêng; API hiện chưa có xác thực nên chỉ dùng URL riêng trong lúc demo.

## Chạy web UI trên máy local

Cần **Node.js 20.19+ hoặc 22.12+** tương thích Vite 7. Laptop không cần GPU.

```bash
git clone https://github.com/nhannhan2201/math-instruct-llama.git
cd math-instruct-llama/frontend
npm ci
npm run dev
```

Mở **http://localhost:5173**. Trong **Cấu hình API**, dán URL gốc ngrok của gateway, ví dụ:

```text
https://your-domain.ngrok-free.dev
```

Không thêm `/docs` hoặc `/solve`. Chọn **LoRA** hoặc **QLoRA**, nhập bài toán rồi gửi. URL và lịch sử hội thoại được lưu trên trình duyệt.

Có thể đặt URL mặc định qua `VITE_API_URL` trong `.env.local`, theo [frontend/.env.example](frontend/.env.example). Nếu đã lưu URL trên UI, cấu hình đó được ưu tiên. Xem thêm [hướng dẫn frontend](frontend/README.md).

## Luồng hoạt động

```text
Web UI · localhost:5173
          ↓ HTTPS qua ngrok
Gateway · Kaggle · port 8080
          ├─ /qlora/solve → API 8000 → QLoRA · GPU 0
          └─ /lora/solve  → API 8001 → LoRA  · GPU 1
```

Gateway chỉ chuyển tiếp request; mỗi API model tải adapter một lần khi khởi động. CORS mặc định cho phép UI ở `localhost:5173` và `127.0.0.1:5173`. Nếu đổi địa chỉ UI, cấu hình `UI_ORIGINS` rồi khởi động lại gateway.

> Model có thể trả lời sai, kể cả bài đơn giản. Hai ảnh trên là minh họa demo, không phải đánh giá accuracy. Mỗi câu hỏi được xử lý độc lập; lịch sử UI không được gửi cho model.

## Chấm tập test MathInstruct

`src/test.py` chấm dự đoán token của lời giải mẫu trên 500 nhóm câu hỏi test đã giữ riêng trong split manifest. Chạy lần lượt trên một GPU Kaggle sau khi cài `requirements.txt` và thêm hai adapter làm Input:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.test --method base
CUDA_VISIBLE_DEVICES=0 python -m src.test --method lora --adapter-path /kaggle/input/datasets/nhannhan22/models/lora_training_5c645da682df/final
CUDA_VISIBLE_DEVICES=0 python -m src.test --method qlora --adapter-path /kaggle/input/datasets/nhannhan22/models/qlora_training_6c775f8d8113/final
```

Thay đường dẫn adapter nếu Kaggle Input của bạn khác. Mỗi lệnh tạo một run trong experiment `llama-3-math-instruct-test` ở `mlruns.db`, ghi `test_loss`, `test_mean_token_accuracy`, tốc độ đánh giá và `test_protocol.json`. Tải database **và thư mục artifact MLflow** từ Kaggle nếu muốn xem đầy đủ trên máy local. Evaluation giữ đầy đủ 500 lời giải, không cắt về giới hạn 1024 token của training; cả ba model dùng cùng protocol. Script dừng nếu snapshot, metadata hoặc độ dài vượt context của model. Token accuracy theo TRL 1.3 là trung bình accuracy từng mẫu (eval batch size 1), được lấy từ callback logging. Đây là teacher-forced token accuracy, **không phải tỷ lệ giải đúng bài toán**.
