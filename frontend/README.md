# Math Studio frontend

React + Vite + Tailwind UI gọi gateway, không tải model.

```bash
cd frontend
npm ci
npm run dev
```

Mở http://localhost:5173. Vite giữ cố định port này; nếu port bị chiếm, dừng ứng dụng khác trước khi chạy. Gateway mặc định cho phép CORS từ localhost:5173 và 127.0.0.1:5173.

Trong hộp Cấu hình API, nhập URL gốc ngrok trỏ tới gateway port 8080 (không thêm /docs). URL được lưu trên trình duyệt. Hoặc sao chép `.env.example` thành `.env.local`, đặt `VITE_API_URL`, rồi chạy lại Vite. Cấu hình đã lưu trên trình duyệt được ưu tiên hơn biến môi trường; thay bằng hộp cấu hình khi cần.

Kaggle phải giữ hai server model, gateway và tunnel đang chạy. UI gọi POST /qlora/solve hoặc /lora/solve và gửi header ngrok-skip-browser-warning. Nếu đổi port/domain của UI, đặt UI_ORIGINS trên gateway và khởi động lại gateway.

Hội thoại lưu trong localStorage (tối đa 30 hội thoại). Chỉ câu hỏi hiện tại được gửi cho model, không có bộ nhớ hội thoại ở backend. Dừng chờ chỉ hủy request từ UI; không đảm bảo dừng generation trên GPU. Markdown và công thức có delimiter `$...$`, `$$...$$` được hiển thị; UI không tự chuyển mọi phép tính dạng text sang LaTeX.

```bash
npm run build
```

Build nằm trong dist/. Không lưu authtoken ngrok hoặc khóa bí mật trong frontend: mọi biến VITE_* đều công khai trong bundle. API hiện chưa có xác thực; bản này phục vụ demo với backend hiện tại.
