# Math Instruct Llama

Fine-tune Llama 3.2 1B Instruct trên MathInstruct bằng LoRA/QLoRA, với deterministic group split, chat completion-only loss và MLflow tracking.

- [GUIDE](docs/GUIDE.md): kiến trúc, preparation/tests, Kaggle smoke và artifacts.
- [PROGRESS](docs/PROGRESS.md): approvals, kết quả thực tế và giới hạn kiểm chứng.
- [Kế hoạch lịch sử](docs/IMPLEMENTATION_PLAN.md).

CPU offline tests và real-data integration đã PASS. Notebook Kaggle xác nhận LoRA và QLoRA đều train đúng hai optimizer steps, validation và save trên Tesla T4 ở commit `905673f`. QLoRA reload/inference PASS; LoRA reload PASS với artifact trước đó, artifact LoRA mới nhất còn cần reload. Chưa chạy full training hoặc benchmark. Handoff mới nhất ở section 15 của PROGRESS.

Với Python và torch/CUDA phù hợp đã provision, chạy từ terminal:

```bash
git clone https://github.com/nhannhan2201/math-instruct-llama.git
cd math-instruct-llama
python -m pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0 python -m src.train --method lora --smoke
```

Training tự load pinned tokenizer/dataset/base model, tạo hoặc kiểm tra manifest,
chạy đúng hai optimizer steps, final validation và lưu artifact vào output riêng.
Lần đầu cần Internet/model access; lần sau dùng cache. Không cần preparation riêng để chạy smoke. Kaggle đã thử cần gỡ `torchao` cũ;
`pip check` còn conflicts với packages có sẵn, nên chưa xác nhận clean setup
chỉ bằng clone/install. Cấu hình, workaround và các tests ở GUIDE.
