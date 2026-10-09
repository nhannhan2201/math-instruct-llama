# Math Instruct Llama

Fine-tune Llama 3.2 1B Instruct trên MathInstruct bằng LoRA/QLoRA, với deterministic group split, chat completion-only loss và MLflow tracking.

- [GUIDE](docs/GUIDE.md): kiến trúc, preparation/tests, Kaggle smoke và artifacts.
- [PROGRESS](docs/PROGRESS.md): approvals, kết quả thực tế và giới hạn kiểm chứng.
- [Kế hoạch lịch sử](docs/IMPLEMENTATION_PLAN.md).

CPU offline tests và real-data integration đã PASS. Smoke mode giới hạn hai optimizer steps và output riêng; GPU/CUDA, real-model save/reload, benchmark và container runtime còn cần xác minh. Chưa chạy GPU/full training trong implementation session.
