"""Fine-tune DeBERTa-v3 cho NER (token classification, 21 nhãn IOB2).

Đọc:  data/labeled/ner_iob2/*.jsonl   (do scripts.convert_spans_to_iob2 sinh ra)
Ghi:  models/deberta-ner-incident/    (model + tokenizer, dùng cho Layer 2)
      models/deberta-ner-incident/eval_metrics.json

Quy trình:
  1. Cross-validation theo TÀI LIỆU (không theo chunk, tránh rò rỉ) với số epoch cố định,
     ghi F1 (seqeval, strict IOB2) của từng epoch trên tập giữ lại.
  2. Chọn số epoch có F1 trung bình các fold cao nhất.
  3. Train lại model cuối trên TOÀN BỘ dữ liệu với số epoch đó rồi lưu.

Lưu ý: nhãn đầu vào là nhãn silver của Qwen, nên F1 đo mức độ model bắt chước Qwen,
không phải độ chính xác thật so với nhãn chuẩn của người.

Chạy:  python -m scripts.train_ner
Nhanh: python -m scripts.train_ner --skip-cv --epochs 8      (bỏ cross-validation)
Nên tắt Ollama trước để nhường VRAM:  docker compose stop ollama
"""

import argparse
import json
import math
import random
import statistics
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from seqeval.metrics import classification_report, f1_score
from seqeval.scheme import IOB2
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.iob2 import IGNORE_INDEX, LABEL2ID
from src.ner_data import ID2LABEL, decode_tags, load_iob2_rows, make_document_folds

NOTE = (
    "Nhãn train là nhãn silver do Qwen sinh, chưa được người review: F1 dưới đây đo mức độ "
    "model bắt chước Qwen, không phải độ chính xác so với nhãn chuẩn."
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    longest = max(len(row["input_ids"]) for row in batch)
    input_ids, attention_mask, labels = [], [], []
    for row in batch:
        pad = longest - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_id] * pad)
        attention_mask.append([1] * len(row["input_ids"]) + [0] * pad)
        labels.append(row["labels"] + [IGNORE_INDEX] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
    }


def build_model(model_name: str, device: torch.device):
    # dtype=float32: transformers 5.x mặc định nạp đúng kiểu lưu trong checkpoint (fp16),
    # mà GradScaler yêu cầu trọng số fp32 (phần tính fp16 do autocast đảm nhiệm).
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(LABEL2ID), id2label=ID2LABEL, label2id=LABEL2ID, dtype=torch.float32
    )
    return model.to(device)


def build_optimizer(model, lr: float, head_lr: float, weight_decay: float):
    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
    groups = []
    for is_head in (False, True):
        for decay in (True, False):
            params = [
                p
                for n, p in model.named_parameters()
                if n.startswith("classifier") == is_head and (not any(nd in n for nd in no_decay)) == decay
            ]
            if params:
                groups.append(
                    {"params": params, "lr": head_lr if is_head else lr, "weight_decay": weight_decay if decay else 0.0}
                )
    return torch.optim.AdamW(groups)


def train_epoch(model, loader, optimizer, scheduler, scaler, device, use_amp) -> float:
    model.train()
    total, steps = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            loss = model(**batch).loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total += loss.item()
        steps += 1
    return total / max(steps, 1)


@torch.no_grad()
def predict(model, rows, pad_id, device, batch_size, use_amp) -> tuple[list[list[str]], list[list[str]]]:
    model.eval()
    loader = DataLoader(rows, batch_size=batch_size, shuffle=False, collate_fn=partial(collate, pad_id=pad_id))
    true_seqs, pred_seqs = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        for labels, preds in zip(batch["labels"].tolist(), logits.argmax(-1).tolist()):
            true_tags, pred_tags = decode_tags(labels, preds)
            true_seqs.append(true_tags)
            pred_seqs.append(pred_tags)
    return true_seqs, pred_seqs


def f1(true_seqs, pred_seqs) -> float:
    return float(f1_score(true_seqs, pred_seqs, mode="strict", scheme=IOB2, zero_division=0))


def fit(model, train_rows, epochs, args, pad_id, device, use_amp, val_rows=None, tag="") -> dict:
    """Train `epochs` epoch trên train_rows; nếu có val_rows thì chấm F1 sau mỗi epoch."""
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        train_rows, batch_size=args.batch_size, shuffle=True, generator=generator,
        collate_fn=partial(collate, pad_id=pad_id),
    )
    optimizer = build_optimizer(model, args.lr, args.head_lr, args.weight_decay)
    total_steps = epochs * math.ceil(len(train_rows) / args.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {"train_loss": [], "val_f1": [], "val_preds": []}
    for epoch in range(1, epochs + 1):
        started = time.time()
        loss = train_epoch(model, loader, optimizer, scheduler, scaler, device, use_amp)
        history["train_loss"].append(loss)
        message = f"  {tag} epoch {epoch:>2}/{epochs}  loss={loss:.4f}"
        if val_rows is not None:
            true_seqs, pred_seqs = predict(model, val_rows, pad_id, device, args.batch_size * 2, use_amp)
            score = f1(true_seqs, pred_seqs)
            history["val_f1"].append(score)
            history["val_preds"].append((true_seqs, pred_seqs))
            message += f"  val_f1={score:.4f}"
        print(f"{message}  ({time.time() - started:.0f}s)", flush=True)
    return history


def cross_validate(rows, args, tokenizer, device, use_amp) -> dict:
    doc_ids = [r["incident_id"] for r in rows]
    folds = make_document_folds(doc_ids, args.folds, args.seed)
    print(f"Cross-validation {len(folds)} fold theo tài liệu: {folds}", flush=True)

    fold_histories = []
    for i, val_docs in enumerate(folds):
        train_rows = [r for r in rows if r["incident_id"] not in val_docs]
        val_rows = [r for r in rows if r["incident_id"] in val_docs]
        print(f"[fold {i + 1}/{len(folds)}] train={len(train_rows)} chunk, val={len(val_rows)} chunk ({val_docs})", flush=True)
        set_seed(args.seed + i)
        model = build_model(args.model, device)
        history = fit(model, train_rows, args.epochs, args, tokenizer.pad_token_id, device, use_amp, val_rows, f"fold {i + 1}")
        history["val_docs"] = val_docs
        fold_histories.append(history)
        del model
        torch.cuda.empty_cache()

    mean_curve = [statistics.mean(h["val_f1"][e] for h in fold_histories) for e in range(args.epochs)]
    best_epoch = int(np.argmax(mean_curve)) + 1
    at_best = [h["val_f1"][best_epoch - 1] for h in fold_histories]

    # Gộp dự đoán ngoài-fold của mọi tài liệu ở epoch tốt nhất để có F1 theo từng loại entity.
    pooled_true = [seq for h in fold_histories for seq in h["val_preds"][best_epoch - 1][0]]
    pooled_pred = [seq for h in fold_histories for seq in h["val_preds"][best_epoch - 1][1]]
    report = classification_report(
        pooled_true, pooled_pred, mode="strict", scheme=IOB2, output_dict=True, zero_division=0
    )
    per_type = {
        name: {k: float(v) for k, v in stats.items()} for name, stats in report.items() if isinstance(stats, dict)
    }

    return {
        "folds": [{"val_docs": h["val_docs"], "f1_per_epoch": h["val_f1"], "train_loss_per_epoch": h["train_loss"]} for h in fold_histories],
        "mean_f1_per_epoch": mean_curve,
        "best_epoch": best_epoch,
        "f1_at_best_epoch": {
            "mean": statistics.mean(at_best),
            "std": statistics.pstdev(at_best),
            "per_fold": at_best,
        },
        "per_type_out_of_fold": per_type,
    }


def main() -> None:
    sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--iob2-dir", type=Path, default=Path("data/labeled/ner_iob2"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/deberta-ner-incident"))
    parser.add_argument("--model", default="microsoft/deberta-v3-base")
    parser.add_argument("--epochs", type=int, default=12, help="số epoch tối đa để thử trong cross-validation (hoặc số epoch train cuối nếu --skip-cv)")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5, help="learning rate phần encoder")
    parser.add_argument("--head-lr", type=float, default=3e-4, help="learning rate lớp phân loại mới")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-cv", action="store_true", help="bỏ cross-validation, train thẳng --epochs epoch")
    parser.add_argument("--fp32", action="store_true", help="tắt fp16 (chậm và tốn VRAM hơn nhưng ổn định hơn)")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    rows = load_iob2_rows(args.iob2_dir)
    if not rows:
        sys.exit(f"Không có dữ liệu trong {args.iob2_dir}/ — chạy python -m scripts.convert_spans_to_iob2 trước.")
    n_docs = len({r["incident_id"] for r in rows})

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB)")
    elif args.allow_cpu:
        device = torch.device("cpu")
    else:
        sys.exit("Không thấy GPU CUDA (torch bản CPU hoặc thiếu driver). Train trên CPU rất chậm; thêm --allow-cpu nếu vẫn muốn.")
    use_amp = device.type == "cuda" and not args.fp32
    print(f"{len(rows)} chunk, {n_docs} tài liệu | fp16={use_amp}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "note": NOTE,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "data": {"chunks": len(rows), "documents": n_docs},
    }

    final_epochs = args.epochs
    if args.skip_cv or n_docs < 2:
        print("Bỏ qua cross-validation.", flush=True)
    else:
        cv = cross_validate(rows, args, tokenizer, device, use_amp)
        metrics["cv"] = cv
        final_epochs = cv["best_epoch"]
        best = cv["f1_at_best_epoch"]
        print(f"\nCV: epoch tốt nhất = {final_epochs}, F1 = {best['mean']:.4f} ± {best['std']:.4f} (từng fold: {[round(x, 3) for x in best['per_fold']]})", flush=True)

    print(f"\nTrain model cuối trên toàn bộ dữ liệu, {final_epochs} epoch", flush=True)
    set_seed(args.seed)
    model = build_model(args.model, device)
    history = fit(model, rows, final_epochs, args, tokenizer.pad_token_id, device, use_amp, None, "final")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    metrics["final"] = {"epochs": final_epochs, "train_loss_per_epoch": history["train_loss"], "output_dir": str(args.output_dir)}

    with (args.output_dir / "eval_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Đã lưu model + eval_metrics.json vào {args.output_dir}/")


if __name__ == "__main__":
    main()
