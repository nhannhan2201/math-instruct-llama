# Math Instruct Llama

Dự án học ML Engineering: fine-tune Llama 3.2 1B Instruct trên MathInstruct bằng LoRA/QLoRA, so sánh với base model, tracking bằng MLflow và hướng tới API/UI/Docker local.

- [GUIDE — kiến trúc, pipeline thực tế, cấu hình và cách kiểm chứng](docs/GUIDE.md).
- [PROGRESS — roadmap, approvals, kết quả và task tiếp theo](docs/PROGRESS.md).
- [Kế hoạch lịch sử — giữ nguyên bằng chứng và diễn tiến](docs/IMPLEMENTATION_PLAN.md).

Data preparation và phân tích tokenizer/collator đã có bằng chứng CPU. Training GPU, chất lượng benchmark, inference/API/Docker runtime chưa được kiểm chứng đầy đủ; UI chưa implement. Chưa tuyên bố production-ready.

Task 3A đã hoàn tất phân tích; chat format, max length và truncation policy đang chờ review. Task 3B chưa được phép triển khai.
