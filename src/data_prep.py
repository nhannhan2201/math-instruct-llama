from datasets import load_dataset
from src.config import (
    DATASET_ID, MAX_EVAL_SAMPLES, TRAIN_FRACTION, 
    SEED, PROMPT_TEMPLATE
)

def format_batch(examples, tokenizer):
    """
    Format dataset theo cấu trúc Prompt Template đã định nghĩa.
    """
    texts = []
    for q, a in zip(examples["instruction"], examples["output"]):
        q = (q or "").strip()
        a = (a or "").strip()
        text = PROMPT_TEMPLATE.format(question=q, answer=a) + tokenizer.eos_token
        texts.append(text)
    return {"text": texts}

def prepare_data(tokenizer):
    """
    Tải, chia split, lấy sample và map dataset với format function.
    """
    raw_ds = load_dataset(DATASET_ID)["train"]
    splits = raw_ds.train_test_split(test_size=MAX_EVAL_SAMPLES, seed=SEED)
    
    train_raw = splits["train"]
    eval_raw = splits["test"]

    # Lấy một phần nhỏ dữ liệu để train (theo TRAIN_FRACTION)
    train_n = int(len(train_raw) * TRAIN_FRACTION)
    train_raw = train_raw.shuffle(seed=SEED).select(range(train_n))

    print(f"Đã chuẩn bị xong MathInstruct: train={len(train_raw)} mẫu, eval={len(eval_raw)} mẫu")

    # Xử lý text
    train_ds = train_raw.map(
        lambda x: format_batch(x, tokenizer), 
        batched=True, 
        remove_columns=train_raw.column_names
    )
    eval_ds = eval_raw.map(
        lambda x: format_batch(x, tokenizer), 
        batched=True, 
        remove_columns=eval_raw.column_names
    )
    
    return train_ds, eval_ds