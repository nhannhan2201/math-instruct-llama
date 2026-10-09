> **TÀI LIỆU LỊCH SỬ — 2026-10-09.** Được thay thế bởi [GUIDE.md](GUIDE.md) (kỹ thuật hiện tại) và [PROGRESS.md](PROGRESS.md) (roadmap/quy trình/approvals). Không tiếp tục cập nhật tài liệu này. Toàn bộ nội dung bên dưới giữ nguyên; trạng thái và hướng dẫn cũ chỉ có ý nghĩa tại thời điểm ghi nhận.

# Math Instruct Llama — Implementation Plan

Ngày cập nhật: **2026-10-09**. Tài liệu này duy trì scope, quyết định và tiến độ giữa các phiên Codex.

## 1. Project Overview

Fine-tune **Llama 3.2 1B Instruct** (`unsloth/Llama-3.2-1B-Instruct`) trên MathInstruct bằng **LoRA và QLoRA**, so sánh với base model chưa fine-tune về answer accuracy, validation loss, training time, peak GPU VRAM và trainable parameters. Dùng MLflow để tracking; FastAPI và Docker để serving model được chọn.

- Reuse code hiện tại; thay đổi từng bước nhỏ, phục vụ học ML Engineering.
- TinyLoRA nằm ngoài phạm vi thực nghiệm; chưa cần xóa code. Chỉ sửa liên quan nếu nó chặn LoRA/QLoRA.
- Không thêm Registry, public deployment, Kubernetes hoặc orchestration. Không rewrite/refactor rộng chỉ để đổi scope.
- Chưa có kết quả benchmark được kiểm chứng; không kết luận phương pháp tốt hơn trước khi đo.

### Quy tắc cho các phiên Codex sau

1. Luôn đọc file này trước khi đề xuất hoặc thực hiện công việc.
2. Chỉ làm một task nhỏ tại một thời điểm; nêu mục tiêu và phạm vi trước khi sửa code.
3. Chỉ triển khai phần được người dùng đồng ý; không tự chuyển sang task tiếp theo.
4. Chạy kiểm tra phù hợp sau task; cập nhật tracker bằng commands và actual results. Chỉ DONE khi acceptance criteria đã được kiểm chứng.
5. Ghi blocker và hướng giải quyết; không suy diễn kết quả hoặc tự cài/tải/chạy training ngoài phạm vi được duyệt.

## 2. Current Implementation Status

Ý nghĩa: **EXISTS** = có code/artifact; **VERIFIED** = đã kiểm tra với phạm vi và bằng chứng rõ; **NOT VERIFIED** = hành vi chưa được xác nhận; **PLANNED** = chưa triển khai. EXISTS không đồng nghĩa DONE.

| Thành phần | Trạng thái và bằng chứng hiện tại |
|---|---|
| Config | EXISTS; VERIFIED qua đọc `src/config.py`: seed 42, eval 100, train fraction 0.03, max length 128. 128 là cấu hình hiện tại, chưa phải lựa chọn cuối. |
| Data preparation | EXISTS; VERIFIED qua đọc `src/data_prep.py`: split eval 100, sample 3% phần còn lại; ghép instruction/output/EOS thành text, bỏ cột nguồn. VERIFIED pipeline Task 2 trên cache: 262.039 raw → 246.002 distinct pairs; subset 7.359 train, 100 validation primary, 500 test groups; manifest verified. Counts Task 1 là pipeline cũ. |
| Tokenization/labels | EXISTS qua SFTTrainer trong `src/train.py`; truncation và labels thực tế NOT VERIFIED. Chưa chỉ định completion-only loss. Inspection EXISTS trong src/inspect_data.py; VERIFIED mô phỏng tokenization/truncation trên toàn bộ 7.858 train samples thật. VERIFIED actual labels trong env Conda project TRL 1.3.0: full-sequence targets, prompt không mask, padding -100; chưa chạy forward/training. |
| LoRA/QLoRA training | EXISTS; LoRA rank 4, alpha 8, dropout 0.05; QLoRA NF4 + double quantization. Training, gradients và reload NOT VERIFIED. TinyLoRA vẫn có trong CLI/import nhưng ngoài scope. |
| Checkpoint | EXISTS: adapter weights/config, tokenizer, chat template, training_args và model card trong `models/qlora_final/`. VERIFIED sự tồn tại; provenance, training completion và chất lượng NOT VERIFIED. |
| MLflow | EXISTS: SQLite URI, log tay method/LR/epochs/batch, report_to=mlflow. Run thực tế, metrics và artifacts NOT VERIFIED. |
| Evaluation | EXISTS: validation định kỳ mỗi 40 steps; chưa có final evaluation bắt buộc. Held-out test, answer accuracy và base baseline PLANNED. |
| Inference/API | EXISTS: MathSolver nạp base + adapter; generation sampling; API cố định QLoRA. Runtime NOT VERIFIED; health hiện trả OK cả khi solver load thất bại. |
| Docker/dependencies | EXISTS; requirements chưa pin, Docker copy models và cài web dependencies trùng. Build/GPU serving NOT VERIFIED. |
| Tests/portfolio | Test suite và root README PLANNED; model-card README cục bộ còn placeholders. |

Kiểm tra 2026-10-09: toàn bộ Python source parse được bằng `ast.parse` (chỉ cú pháp); Python hiện tại thiếu `peft`, `trl`, `bitsandbytes`, `mlflow`. Không import training/API, không load model hoặc deserialize `training_args.bin`.

## 3. Implementation Roadmap

Các đường dẫn file mới dưới đây là dự kiến, chưa tồn tại. Thứ tự task là dependency mặc định; hoàn tất task không tự cấp quyền làm task tiếp theo.

### Task 1 — Tokenization, truncation và labels inspection

- **Mục tiêu:** hiểu model nhìn thấy gì và vùng nào chịu loss trước khi training.
- **Files:** `src/inspect_data.py` riêng; `src/data_prep.py` chỉ chuẩn bị dữ liệu. Reuse `prepare_data()` và `format_batch()`; không duplicate split/sampling hoặc thay đổi hai functions cũ.
- **Deliverables:** inspection toàn bộ train MathInstruct thật từ cache; min/median/p90/p95/p99/max, tỷ lệ vượt 128/256/512, answer truncation/EOS/no-answer counts, source distribution và 3–5 ví dụ thật. Ghi revision từ cache; không dùng synthetic fixtures.
- **Acceptance:** load tokenizer với `local_files_only=True`, không tải dataset/model; reuse formatter; dùng offset mappings xác định answer span. Labels mô phỏng toàn chuỗi/mask prompt được phân biệt rõ với labels từ SFT collator. Thiếu TRL thì ghi NEEDS VERIFICATION; phần kiểm chứng labels thực tế chưa đạt, không đánh dấu toàn task DONE. Có cảnh báo answer bị cắt/mất hoàn toàn; padding labels là -100.
- **Kiến thức:** token IDs, causal next-token prediction, truncation, attention mask, labels -100 và prompt masking. Kết quả chỉ đại diện train subset hiện tại, không phải toàn bộ MathInstruct; tokenizer-level simulation chưa chứng minh SFTTrainer preprocessing.

### Task 2 — Data split, deduplication và manifest

- **Mục tiêu:** split tái lập được và hạn chế leakage theo nội dung.
- **Files:** `src/data_prep.py`, `src/config.py`; manifest/report trong `data/`.
- **Deliverables:** dedup normalized question/output pair, giữ distinct outputs; split theo question groups; primary/all references tách riêng; lưu raw indices/source, revision, seed và hashes. Không tự loại outputs khác nhau vì chưa chứng minh final-answer conflict.
- **Acceptance:** không trùng normalized question giữa splits; cùng snapshot/seed cho cùng manifest; báo actual counts sau lọc, split và sampling.
- **Kiến thức:** duplicates, leakage, dataset revision và reproducibility.

### Task 3 — Training formatting và completion-only loss

- **Mục tiêu:** học lời giải với định dạng nhất quán và không mất answer âm thầm.
- **Files:** data/config/training hiện tại; formatter dùng chung với `src/inference.py`.
- **Deliverables:** prompt/completion format, chat formatter chung với system instruction cố định; explicit completion-only loss; báo cáo token lengths trên dữ liệu thật và đề xuất max length để review.
- **Acceptance:** kiểm tra labels thực tế: prompt/padding -100, mỗi training sample có answer tokens chịu loss; báo mẫu bị cắt/loại và EOS. Chốt sequence length sau dữ liệu và ngân sách VRAM, trước full training.
- **Kiến thức:** chat template, completion masking, EOS và phân bố độ dài.

### Task 4 — Evaluation và base baseline

- **Mục tiêu:** đo đáp án đúng, không chỉ loss.
- **Files:** thêm `src/evaluate.py`, tests cho evaluator; mở rộng inference hỗ trợ base-only và greedy generation.
- **Deliverables:** extraction/normalization cho numeric/multiple-choice; predictions, eligible IDs, coverage, accuracy và base validation loss.
- **Acceptance:** fixtures kiểm tra fraction/decimal/choice và extraction failures; không lấy số cuối tùy tiện. Reference không hỗ trợ loại trước benchmark; prediction không trích được tính sai trên eligible subset. Base dùng cùng evaluation protocol.
- **Kiến thức:** accuracy vs loss, normalization, denominator và giới hạn exact match.

### Task 5 — LoRA/QLoRA smoke training

- **Mục tiêu:** chứng minh pipeline chạy trước khi đầu tư full training.
- **Files:** `requirements.txt`, `src/train.py`, `src/config.py`; chỉ sửa TinyLoRA import nếu gây blocker.
- **Deliverables:** bộ dependency versions đã kiểm chứng; hai optimizer steps mỗi method trên subset nhỏ; save/reload adapter.
- **Acceptance:** GPU/environment được xác nhận; finite loss, adapter có gradients, base frozen; reload và generation thành công. Seed đặt trước adapter initialization, precision/checkpointing rõ ràng. Chỉ chạy sau khi được duyệt.
- **Kiến thức:** PEFT, LoRA A/B, NF4, frozen weights và dependency compatibility.

### Task 6 — MLflow và resource metrics

- **Mục tiêu:** experiment có thể truy nguyên và so sánh chi phí.
- **Files:** `src/train.py`, config; helper tracking nhỏ nếu cần.
- **Deliverables:** log effective config, revisions/split hash, adapter settings, actual counts, loss/LR, final validation, elapsed time, CUDA peaks, trainable count; config/data report/adapter metadata artifacts.
- **Acceptance:** smoke run có dữ liệu trong đúng MLflow run; final validation luôn chạy; không coi console print hoặc local save là MLflow logging. Phân biệt Trainer metrics với measurements tự đo.
- **Kiến thức:** parameters/metrics/artifacts, CUDA allocated/reserved, synchronization và timing boundaries.

### Task 7 — Controlled training và benchmark

- **Mục tiêu:** so sánh base/LoRA/QLoRA bằng bằng chứng.
- **Files:** training/evaluator hiện có; benchmark report và README dự kiến.
- **Deliverables:** runs được duyệt, bảng năm metrics mục tiêu, predictions và review 20 lỗi; base NF4 là đối chứng phụ nếu khả thi.
- **Acceptance:** cùng manifest, evaluation/generation settings và hardware; trace được từng hàng về config/run/artifact; báo thiếu metrics và giới hạn một seed, không bịa số hoặc tuyên bố tối ưu. Test chỉ dùng sau khi khóa cấu hình.
- **Kiến thức:** controlled comparison, quantization confound, trade-off và error analysis.

### Task 8 — FastAPI, Docker và portfolio

- **Mục tiêu:** serving adapter được chọn, chạy local và trình bày project tái lập được.
- **Files:** `src/inference.py`, `app.py`, Docker/dependencies, API tests và root `README.md`.
- **Deliverables:** adapter/method configurable; tokenizer từ artifact; lifespan/readiness; request bounds, lỗi client gọn, serialize generation; Docker và hướng dẫn GPU/cache/adapter provisioning.
- **Acceptance:** mocked API tests cho valid/invalid request, unavailable model và generation error; readiness 503 khi chưa sẵn sàng; local CLI/API và Docker GPU inference thực tế chạy. Không đánh dấu runtime VERIFIED chỉ từ mocks.
- **Kiến thức:** Pydantic, model lifecycle, readiness, concurrency và GPU containers.

## 4. Benchmark Protocol

- Seed **42**; deduplicate trước split. Validation **100 question groups**, held-out test **500 question groups**; train subset = **3% records của train pool sau dedup và group split**, lấy floor. Mỗi evaluation group có primary reference và all references riêng. Lưu actual counts và manifest, không suy ra số đã dùng từ dataset card.
- Chung dataset snapshot/splits, formatter, tokenizer policy, completion mask, evaluation và hardware. Validation dùng để điều chỉnh; test khóa trước benchmark.
- Bảng chính: base chưa fine-tune, LoRA, QLoRA. Base NF4 là đối chứng phụ nếu khả thi để tách ảnh hưởng quantization; nếu bỏ phải ghi lý do.
- Hiện code max length 128; **chưa chốt max sequence length cuối cùng**. Task 1/3 cung cấp bằng chứng truncation trước quyết định.
- Điểm xuất phát: một epoch, microbatch 1, accumulation 32, LR 2e-4; LoRA/QLoRA cùng rank 4, alpha 8, dropout 0.05 và target modules hiện tại. Khóa effective config trước training; không dựa vào library defaults cho precision/checkpointing.
- Answer accuracy: eligible numeric/multiple-choice subset; normalize fraction/finite decimal/choice, không symbolic equivalence v1. Công bố eligible count/coverage và output bị cắt. Generation chung: `do_sample=False`, max_new_tokens 512, cùng stop tokens.
- Validation loss: cùng validation examples, preprocessing và loss mask; final evaluation bắt buộc.
- Training time: từ trước `trainer.train()` tới sau final save, đồng bộ CUDA ở biên; load/preparation đo riêng. Reset peak CUDA stats ở đầu cùng phạm vi; báo peak allocated/reserved và GPU identity.
- Trainable parameters: đếm unique parameters có `requires_grad=True` trước training. Base có 0 trainable parameters trong benchmark; training time/VRAM training là N/A. Không trộn inference VRAM vào training VRAM.
- Chưa có actual benchmark results. Một seed/cấu hình là kết quả portfolio ban đầu, không chứng minh phương pháp hoặc hyperparameters tối ưu.

## 5. Progress Tracker

Trạng thái: **TODO / IN PROGRESS / DONE / BLOCKED**. EXISTS không tự thay đổi tracker. DONE cần tất cả acceptance criteria; blocker phải ghi bằng chứng và hướng giải quyết.

| Task | Status | Actual results / unresolved issues |
|---|---|---|
| 1. Inspection | DONE | PASS: real-data statistics và actual SFTTrainer preprocessing/default collator trên 3 samples CPU, TRL 1.3.0; giới hạn random model/no forward được ghi rõ. Chờ review trước Task 2. |
| 2. Split/manifest | DONE | PASS: dedup 16.037 records, group splits 100 validation/500 test, subset 7.359; manifest identity/integrity và hai real runs tái lập, zero normalized-question overlap. |
| 3. Formatting/loss | IN PROGRESS | Task 3A DONE: phân tích 7.359 train/100 validation primary, hai formats, năm ngưỡng; actual labels CPU PASS. Chờ review; Task 3B chưa triển khai, max length cuối chưa chốt. |
| 4. Evaluation/base | TODO | Chưa có evaluator hoặc baseline metrics. |
| 5. Smoke training | TODO | Dependencies và GPU environment chưa được kiểm chứng đầy đủ. |
| 6. MLflow/resources | TODO | Logging code có; actual run/resource measurements chưa kiểm chứng. |
| 7. Benchmark | TODO | Chưa có controlled runs/kết quả. |
| 8. Serving/portfolio | TODO | API/Docker có; runtime/tests/root README chưa hoàn tất. |

### Update record — 2026-10-09

- **Scope:** chỉ tạo tài liệu roadmap; không triển khai Task 1.
- **Files changed:** `docs/IMPLEMENTATION_PLAN.md`.
- **Verification commands:** `git status --short`; `rg --files --hidden` (loại .git/models/cache); đọc source/config/Docker/requirements; Python `ast.parse` toàn bộ source, `importlib.metadata.version` cho peft/trl/bitsandbytes/mlflow, liệt kê artifact cục bộ; kiểm tra tài liệu và `git diff --check` sau tạo.
- **Actual results:** source parse thành công; bốn dependencies trên chưa có trong current Python; adapter/tokenizer files tồn tại; không có root README. Document checks PASS: đủ bảy phần, tám tasks TODO, không trailing whitespace. `git diff --check` không báo lỗi (file mới chưa tracked); `git status --short` chỉ có `?? docs/`, chứa file kế hoạch này.
- **Unresolved issues:** labels/truncation thực tế, provenance adapter, GPU/runtime và các metrics chưa kiểm chứng. Hướng giải quyết: Task 1 trước; kiểm chứng dependencies/GPU ở Task 5 sau khi được duyệt. Không tự cài package.

Các lần cập nhật sau ghi ngày, task, files changed, commands, actual results và unresolved issues; không chép lại toàn bộ audit.

### Task 1 update — 2026-10-09 (lịch sử, synthetic đã bị thay thế)

- **Files changed:** `src/data_prep.py`, `docs/IMPLEMENTATION_PLAN.md`.
- **Verification commands:** `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m src.data_prep`; offline assertions cho retained answer/EOS, prompt/padding labels; AST comparison với HEAD cho hai functions cũ; `git diff --check`.
- **Actual results:** 10 fixtures: 7 answers intact, 1 cut, 1 lost, 1 empty. Sample 8: 403 → 128 tokens, answer 366 → 92, mất EOS; sample 9: 399 → 128 tokens, answer 2 → 0, mất EOS. Cắt bên phải, padding bên trái. Assertions PASS; hai functions cũ không đổi AST. Đây không phải thống kê MathInstruct thật.
- **Blocker:** TRL chưa có; chưa kiểm chứng collator/end-to-end SFTTrainer. Dùng môi trường TRL sẵn có do người dùng chọn hoặc xin phép cài ở bước riêng; không tự cài. Không tải model/dataset hoặc chạy training.

### Task 1 real-data update — 2026-10-09

- **Files changed:** `src/inspect_data.py` (mới), `src/data_prep.py` (gỡ inspection/synthetic entrypoint), file roadmap này. Config/train không đổi; AST hai functions data preparation không đổi so với HEAD.
- **Commands:** `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' python -m src.inspect_data > /tmp/math-real-inspection.txt`; AST comparison; assertions trên JSON report (counts/source sum, formatter, 5 examples và simulated labels); `git diff --check`.
- **Provenance:** dataset revision đọc từ raw Arrow cache path: `b4fdc323a7be1379c9c7c0b67b1de72dfee2111a`. Tokenizer load đúng BASE_MODEL bằng `local_files_only=True`, adjustment pad giống train.py; cắt phải, pad trái, add_special_tokens=True. Không load weights, training hoặc download.
- **Actual results:** 7.858 train samples. Token length min 58, median 192, p90 467, p95 656,15, p99 1.018,86, max 2.012 (percentile linear interpolation). Vượt 128: 7.040 (89,59%); 256: 2.105 (26,79%); 512: 649 (8,26%).
- **Truncation mô phỏng tại 128:** 7.005 answers mất ít nhất một token, gồm 1.107 mất toàn bộ answer; 7.040 mất EOS. Trước cắt không có answer rỗng hoặc EOS thiếu. Không phải labels/preprocessing thực tế của SFTTrainer.
- **Source:** đối chiếu toàn raw dataset bằng formatter, không split/sampling lại. 845 train entries khớp nhiều raw rows; 552 có nhiều source candidates, ghi AMBIGUOUS_SOURCE thay vì đoán. 7.306 entries còn lại có source duy nhất; counts đầy đủ trong report. Ví dụ dùng raw row đại diện và ghi match count.
- **Verification:** assertions PASS cho source totals, 5 real examples (cut/intact/lost), formatter equality và masks mô phỏng; syntax/AST PASS. Không padding trong report examples, labels full-sequence/completion-only chỉ để minh họa.
- **Blocker / unresolved:** TRL NOT INSTALLED; actual collator và end-to-end Trainer labels NEEDS VERIFICATION. Không tự cài. Review mức truncation và cách kiểm chứng labels trước khi thay đổi max length/loss ở task sau.

### Task 1 JSON output update — 2026-10-09

- **Files changed:** `src/inspect_data.py`, `docs/data_inspection_report.json` (generated), roadmap này; bỏ `docs/DATA_INSPECTION_REPORT.md` và Markdown renderer.
- **Decision:** người dùng yêu cầu JSON thay vì Markdown. Script tự serialize report vừa tính trong mỗi lần chạy vào `docs/data_inspection_report.json` (UTF-8, indent=2), ghi đè kết quả cũ; terminal chỉ báo summary/path. Không nhập metrics bằng tay.
- **Command:** `python -m src.inspect_data` (offline CPU). Kiểm tra file bằng `json.load`, sample count/source totals/examples; `git diff --check`.
- **Status:** vẫn chưa DONE vì actual Trainer labels chưa xác minh; không chuyển Task 2.

### Task 1 actual-label verification — 2026-10-09

- **Files changed:** `src/inspect_data.py`, `docs/data_inspection_report.json`, roadmap này. SHA256 trước/sau xác nhận `data_prep.py`, `config.py`, `train.py` không đổi.
- **Command:** `/home/nhan/miniconda3/envs/math-instruct-llama/bin/python -m src.inspect_data --verify-labels`. Script offline CPU; report JSON chi tiết, terminal summary và 3 decoded inputs. Dùng môi trường này, không Python mặc định thiếu TRL.
- **Environment:** TRL 1.3.0, Transformers 5.7.0, PyTorch 2.11.0, PEFT 0.19.1, Accelerate 1.13.0. Nhận định thiếu TRL trước đây chỉ áp dụng Python mặc định; blocker đã được giải quyết bằng env sẵn có, không cài package.
- **Actual evidence:** SFTTrainer thật preprocess 3 samples từ prepare_data với tokenizer Llama cached và formatting callback giống training; lấy batch từ default DataCollatorForLanguageModeling. completion_only_loss=False, assistant_only_loss=False, packing=False, padding_free=False, keep_start/max_length=128. Batch 3×128; prompt không mask; labels bằng input_ids ở mọi token thật, -100 ở padding bên phải. Single-sample collator batches cũng PASS (microbatch training 1).
- **Results sau next-token shift:** train index 3: 79 prompt targets, 33 answer targets, 14 padding; index 0: 117 prompt, 10 answer, 0 padding; index 4: 127 prompt, 0 answer, 0 padding. Sample mất answer vẫn có targets từ prompt. Đây là eligibility từ labels, không phải numerical loss đo bằng forward.
- **Source evidence:** đọc source TRL cài trong env; text-only dataset mặc định full-sequence loss, collator copy input_ids và mask padding. Thực nghiệm batch xác nhận behavior này ở versions đã ghi.
- **Checks:** PASS cho preprocessing vs tokenizer prefix, CPU, padding có mặt/ignored, real labels, prompt unmasked, shifted positions và single-sample consistency; JSON assertions PASS; `git diff --check` PASS.
- **Limits:** model Llama cực nhỏ random weights từ config chỉ để khởi tạo Trainer; không load base/PEFT/quantization weights, không forward/backward/train. Overrides CPU, precision, tracking, optimizer và gradient checkpointing chỉ trong harness; không đổi config training. Không chứng minh chất lượng model/numerical loss; Kaggle phải kiểm tra lại versions. Simulated labels vẫn tách riêng trong examples; actual_labels_verification chứa tensors thật.
- **Status:** Task 1 DONE trong phạm vi data/preprocessing/collator đã kiểm chứng; dừng để review. Task 2 xử lý dedup/split; Task 3 xem completion-only loss và max length sau khi được duyệt, chưa sửa ở lượt này.

## 6. Decision Log

| Ngày | Quyết định | Lý do |
|---|---|---|
| 2026-10-09 | Quyết định mới thay phương án A: chỉ real MathInstruct, inspection tách sang inspect_data.py; gỡ synthetic. | Tách trách nhiệm; dùng đúng tập train từ prepare_data và tokenizer training cached, không duplicate split/sampling. |
| 2026-10-09 | Dataset đã được người dùng cho phép tải vào HF cache mặc định. | Cache thật đã có, inspection hiện chạy offline; không lưu dataset vào Git. |
| 2026-10-09 | Chỉ LoRA/QLoRA và base; TinyLoRA ngoài thực nghiệm, giữ code hiện tại. | Thu hẹp scope và reuse pipeline; tránh thay đổi không phục vụ mục tiêu. |
| 2026-10-09 | Inspection trước training; chưa chốt max length. | 128 tokens có thể cắt answer; phải kiểm chứng labels và truncation trước chi phí GPU. |
| 2026-10-09 | Test riêng, manifest cố định, seed 42 và dedup trước split. | Hạn chế leakage và giúp so sánh tái lập. |
| 2026-10-09 | Base NF4 chỉ là đối chứng phụ nếu khả thi. | Phân biệt ảnh hưởng quantization với fine-tuning mà không mở rộng training scope. |
| 2026-10-09 | MLflow tracking, FastAPI/Docker local; không thêm hạ tầng phức tạp. | Đủ phục vụ portfolio Intern và việc học từng component. |
| 2026-10-09 | Một task nhỏ mỗi lần, chuyển task cần người dùng đồng ý. | Giữ khả năng học, review và kiểm soát phạm vi giữa các phiên. |

## 7. Next Action

**Review kết quả Task 2** trong `docs/data_preparation_report.json` và manifest `data/split_manifest.json`. Task 2 DONE trong phạm vi normalized-question equality; chưa triển khai Task 3. Report Task 1 và duplicate audit cũ là bằng chứng pipeline trước thay đổi, không tự ghi đè.

### Task 2 duplicate audit — 2026-10-09

- **Files changed:** `src/inspect_data.py`, `docs/data_duplicates_report.json` (generated), roadmap này.
- **Scope:** `--check-duplicates` chạy offline CPU; reuse prepare_data và formatter matcher để đối chiếu hai splits hiện tại, không duplicate split/sampling. Chuẩn hóa question/output bằng `" ".join(text.split())`, giữ case, punctuation, ký hiệu toán.
- **Commands:** Conda project Python `-m src.inspect_data --check-duplicates` chạy hai lần; assertions cho duplicate/whitespace/alternative output/symbol/case/overlap; JSON total assertions; SHA256 so sánh source/report Task 1 và hai lần report mới; `git diff --check`.
- **Actual results:** 262.039 raw records; 224.460 unique questions; 246.002 unique question/output pairs. 12.202 duplicate pair groups, 28.239 records trong các groups, 16.037 surplus records nếu mỗi pair giữ một bản. 8.993 questions có nhiều outputs khác nhau. Train 7.858 / validation 100 có 5 shared questions, liên quan 5 train và 5 validation samples.
- **Verification:** targeted assertions và JSON counts PASS; source data_prep/config/train và report Task 1 giữ nguyên. Offline loader cần tạo lock ngoài workspace; chạy với quyền cache sau lỗi filesystem read-only, không download/cài package/load weights/training.
- **Limits / unresolved:** khác output không chứng minh mâu thuẫn final answer; chưa đo near/semantic duplicates hoặc train/test (chưa có test). Raw indices gắn với snapshot cache, không đoán unique raw identity cho prepared samples. Cần review quy tắc dedup/group splitting; chưa triển khai dedup, test split hoặc manifest nên Task 2 chưa DONE.

### Task 2 deterministic preparation — 2026-10-09

- **Files changed:** `src/data_prep.py`, `src/config.py`, `tests/test_data_prep.py`, `tests/verify_real_data.py`, `docs/data_preparation_report.json`, roadmap; generated `data/split_manifest.json` (Git ignored). Không đổi formatter AST, train.py hoặc report Task 1.
- **Decision:** giữ exact normalized pair có index nhỏ nhất, all duplicate indices vẫn truy vết; distinct outputs giữ trong train pool. Group sources bao gồm mọi raw provenance. Evaluation primary là index đại diện nhỏ nhất, all references gồm một index mỗi distinct output. Không tự loại output vì khác lời giải.
- **Stratification evidence:** 13 source-combination strata: sizes [300,417,666,1829,7473,7491,8436,9772,12800,13447,24278,48302,89249]; không stratum dưới 100. Giữ source-combination Hamilton quota với integer remainder/lexical ties; test trước, validation trên phần còn lại. Selection SHA256(seed,purpose,identity); subset quota theo representative source. Nguồn nhỏ có thể quota 0.
- **Actual counts:** 262.039 raw → 246.002 distinct pairs (loại 16.037 duplicates); 224.460 question groups. Train pool 223.860 groups / 245.309 records; validation 100 groups / 108 all-reference records / 100 primary; test 500 groups / 585 all-reference records / 500 primary. Subset floor(0.03×245309)=7.359 records.
- **CoT/PoT records:** train pool 172.721/72.588; subset 5.181/2.178; validation all 77/31, primary 69/31; test all 425/160, primary 346/154. Full source distributions và group-source combinations lưu trong generated JSON.
- **Manifest:** 115.897.944 bytes (~110,5 MiB), compact JSON không full texts; revision b4fdc323a7be1379c9c7c0b67b1de72dfee2111a từ cache. Snapshot hash 6b438786f5ef69c39ac752b4d6eb7ccbfbfd69787ea61047f4a30702a475cc1a. Canonical manifest hash 314b5ab21e9464bd88c76bcb1530a106a3d9d40faa0bb9a8cd37927281e681eb. Reload recompute toàn snapshot/expected manifest; mismatch fail, không overwrite. Path/cache timestamp không nằm trong identity.
- **Commands:** project Conda Python `-m unittest discover -s tests -v`; `PYTHONPATH=. python tests/verify_real_data.py` (harness tương đương chạy từ /tmp khi kiểm chứng); AST comparison format_batch; SHA256 train.py/report Task 1; `git diff --check`. Integration offline CPU, cached tokenizer only; loader cần cache lock permission.
- **PASS:** 7 automated unittest cases; real complete preparation hai lần cho cùng manifest bytes/hash, ordered subset, formatted train/validation và test reference hashes; train/validation/test normalized-question overlap=0; schema text-only và existing prepare_data(tokenizer) signature. Initial integration harness FAIL do datasets.Column không JSON serializable, sửa thành list và rerun hai lần PASS; không phải lỗi pipeline.
- **Risks / limits:** semantic/near duplicates chưa đo; CoT/PoT cùng bài nhưng thêm instruction suffix vẫn là groups khác. Whitespace normalization output bỏ khác biệt indentation của PoT cho mục đích dedup, training giữ raw text đại diện. Nhiều lời giải chưa đảm bảo đúng; primary chọn index không đảm bảo chất lượng. Manifest lớn và full recomputation tốn CPU/RAM; ignored data artifacts cần copy sang Kaggle hoặc tái tạo cùng snapshot. Không training, đổi max length/loss hoặc làm Task 3.

### Task 3A — token lengths, truncation và completion-only analysis — 2026-10-09

- **Authorization/scope:** roadmap tổng thể APPROVED; chỉ Task 3A được triển khai. Không tự chuyển 3B, không sửa production pipeline, split/manifest, tải weights, forward/backward/training hoặc dùng test để chọn cấu hình.
- **Files changed:** thêm `src/analyze_task3a.py`, generated `docs/task3a_analysis_report.json`; cập nhật tracker và record này. Các reports Task 1/2 giữ nguyên.
- **Command:** `PYTHONDONTWRITEBYTECODE=1 /home/nhan/miniconda3/envs/math-instruct-llama/bin/python -B -m src.analyze_task3a > /tmp/task3a-analysis.log 2>&1`. Script tự bật HF_HUB_OFFLINE/HF_DATASETS_OFFLINE, ẩn CUDA, tokenizer local_files_only. Đọc Arrow cache paths từ report Task 1 để tránh cache-lock writes; kiểm tra revision, toàn raw_hashes và snapshot_hash với manifest trước khi chọn indices. Vì vậy phụ thuộc cache paths của report cũ, không phải loader portable cho production.
- **Dataset:** đúng ordered 7.359 training records (CoT 5.181/PoT 2.178) và 100 primary validation references (CoT 69/PoT 31); không tokenize hoặc phân tích test. Manifest SHA256 b83ac64a3ea90090a11b12febba9d645bd2e7c1406eabd282bfa70ca90411647 giữ nguyên.
- **Formats:** current formatter được reuse; candidate conversational prompt/completion dùng cached Llama template, cùng system instruction, fixed date_string `09 Oct 2026`. Template chèn ngày nên phải cố định policy nếu triển khai. Backend tokenizer/template hashes lưu trong report. Tokenizer eos_token là `<|eot_id|>` (128009), pad_token_id 128004; tokenizer padding_side left nhưng collator thực tế pad right.
- **Length evidence:** report có min/median/p90/p95/p99/max của prompt/answer/total, train/validation và CoT/PoT riêng, percentiles NumPy linear. Prompt gồm BOS/header; answer dùng full-sample offset spans. Prompt prefix equality PASS cho mọi selected record, không answer rỗng; một terminal EOS/EOT còn lại mỗi sample. Current train median/p95/p99/max = 192/632/939,84/2.126; chat = 219/659,10/967,42/2.154. Current validation max 898, chat max 926.
- **Truncation evidence:** keep_start tại 128/256/512/1024/2048; counts/rates sample truncated, answer partial/lost và EOS/EOT lost riêng. Current train truncated = 6.596/1.955/599/43/1; answer lost = 1.004/103/6/0/0. Chat train truncated = 7.167/2.458/658/49/1; answer lost = 3.006/138/8/0/0. Chat validation truncated = 99/33/10/0/0; answer lost = 40/1/0/0/0.
- **Actual labels:** TRL 1.3.0/Transformers 5.7.0, tiny random Llama config only, không load pretrained weights hoặc forward. Hai formats × train/validation × năm max lengths × keep_start/keep_end; 9 real train examples và 6 validation examples mỗi format, tổng 300 sample/length/mode cases. SFTTrainer preprocessing full IDs khớp tokenizer; chat completion_mask khớp prompt boundary; collator labels/padding/truncation và microbatch-one consistency PASS. Current control completion_only_loss=False; chat True, assistant_only_loss=False, packing/padding_free=False. Chat prompt/padding -100; answer/EOT active nếu còn trong slice. keep_start có cases mất hết completion; keep_end giữ đuôi nhưng mất context. Actual ValueError xác minh formatting_func incompatible với completion_only_loss=True. Cached template không có generation-mask blocks, chưa chọn assistant_only_loss.
- **Token/padding evidence:** volume và fixed-max/dynamic batch 1/4/8 slots được tính từ CPU lengths; dynamic batches theo manifest order, no packing/pad multiple, không dự báo VRAM/speed GPU. Chat retained token volume = 939.477/1.574.131/1.885.891/2.021.502/2.029.816; full volume 2.029.922. Microbatch 1 dynamic padding = 0 ở mọi ngưỡng; gradient accumulation không pad chung 32 samples.
- **Verification:** report assertions PASS cho split-kind totals, nonempty answers, prefix equality, terminal-token accounting, truncation category bounds, monotonic volumes, dynamic microbatch-one padding zero, actual-label evidence và unchanged manifest hash. `ast.parse` analysis script PASS; SHA256 checks production config/data_prep/train/inference PASS; `git diff --check` PASS. Lượt đầu assertion FAIL do chưa lấy rõ input_ids từ apply_chat_template result, sửa analysis script và rerun PASS; không phải production bug. BOS zero-offset được tính vào prompt, không vào answer.
- **Proposal, chưa áp dụng:** conversational prompt/completion, explicit completion_only_loss=True, formatter chung, fixed template/date policy; giữ toàn question/answer/EOT. Thử 1024 và 2048 trên Kaggle T4; reject/quarantine overlength records với IDs/counts thay vì cắt mất final answer hoặc code. Chat 1024 giữ 7.310 complete records, 2048 giữ 7.358; toàn validation giữ nguyên ở cả hai. Length-filter policy/effective IDs cần review trước 3B; manifest selection không thay đổi.
- **Limits/next:** thống kê truncation là tokenizer arithmetic cho toàn selected dataset; actual Trainer/collator verification trên selected representative samples, không mọi record, không numerical loss/training/VRAM. Actual keep_end chỉ để minh họa, không đề xuất production. Cache portability, Kaggle package/kernel/VRAM và hyperparameters vẫn chưa kiểm chứng. Task 3A DONE; dừng chờ review, Task 3B chưa được triển khai.
