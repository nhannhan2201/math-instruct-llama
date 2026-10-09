# Math Instruct Llama

Fine-tune Llama 3.2 1B Instruct trên MathInstruct bằng LoRA/QLoRA, với deterministic group split, chat completion-only loss và MLflow tracking.

- [GUIDE](docs/GUIDE.md): kiến trúc, preparation/tests, Kaggle smoke và artifacts.
- [PROGRESS](docs/PROGRESS.md): approvals, kết quả thực tế và giới hạn kiểm chứng.
- [Kế hoạch lịch sử](docs/IMPLEMENTATION_PLAN.md).

CPU offline tests và real-data integration đã PASS. Smoke mode giới hạn hai optimizer steps và output riêng; GPU/CUDA, real-model save/reload, benchmark và container runtime còn cần xác minh. Chưa chạy GPU/full training trong implementation session.

Với Python và torch/CUDA phù hợp đã provision, chạy từ terminal:

```bash
git clone https://github.com/nhannhan2201/math-instruct-llama.git
cd math-instruct-llama
python -m pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0 python -m src.train --method lora --smoke
```

Training tự load pinned tokenizer/dataset/base model, tạo hoặc kiểm tra manifest,
chạy đúng hai optimizer steps, final validation và lưu artifact vào output riêng.
Lần đầu cần Internet/model access; lần sau dùng cache. Không cần patch notebook
hoặc preparation riêng để chạy smoke. Cấu hình torch/CUDA và các tests ở GUIDE.
