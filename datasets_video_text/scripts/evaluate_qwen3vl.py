#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path


DEFAULT_PROJECT_ROOT = Path(
    "/XYFS01/HDD_POOL/hitsz_mszhang/hitsz_mszhang_1/MRC/MRC/MRC_project/others/AAA/vlm/Graduation-design"
)
DEFAULT_MODEL_PATH = Path(
    "/XYFS01/HDD_POOL/hitsz_mszhang/hitsz_mszhang_1/MRC/MRC/MRC_project/others/AAA/vlm/hfmodel/qwen3vl_8b"
)


ALIASES = {
    "anger": ["anger", "angry", "ang"],
    "disgust": ["disgust", "disgusted", "dis"],
    "fear": ["fear", "fearful", "scared", "fea"],
    "frustration": ["frustration", "frustrated", "fru"],
    "happiness": ["happiness", "happy", "hap"],
    "joy": ["joy", "happy", "happiness"],
    "neutral": ["neutral", "neu"],
    "sadness": ["sadness", "sad", "sadness.", "sadness,", "sad"],
    "surprise": ["surprise", "surprised", "sur"],
    "excitement": ["excitement", "excited", "exc"],
    "other": ["other", "others", "oth"],
}


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def load_labels(label_file, rows):
    if label_file and Path(label_file).exists():
        data = json.loads(Path(label_file).read_text(encoding="utf-8"))
        labels = data.get("emotion") or data.get("labels")
        if labels:
            return [str(label).lower() for label in labels]
    return sorted({str(row["emotion"]).lower() for row in rows if row.get("emotion")})


def strict_prompt(base_prompt, labels):
    label_text = ", ".join(labels)
    return (
        f"{base_prompt}\n\n"
        "Important output rule: return only one lowercase emotion label from "
        f"this list: {label_text}. Do not include explanation, punctuation, "
        "markdown, or JSON."
    )


def normalize_text(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def maybe_extract_json_label(text):
    text = text.strip()
    candidates = [text]
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if match:
        candidates.insert(0, match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except Exception:
            continue
        if isinstance(value, dict):
            for key in ("emotion", "label", "answer", "prediction"):
                if key in value:
                    return str(value[key])
        elif isinstance(value, str):
            return value
    return None


def extract_label(output_text, labels):
    labels = [label.lower() for label in labels]
    normalized = normalize_text(maybe_extract_json_label(output_text) or output_text)
    compact = normalized.strip(" .,:;\"'`[](){}")
    if compact in labels:
        return compact

    for label in labels:
        aliases = ALIASES.get(label, [label])
        for alias in aliases:
            pattern = r"(?<![a-z])" + re.escape(alias.lower()) + r"(?![a-z])"
            if re.search(pattern, normalized):
                return label
    return None


def weighted_f1(y_true, y_pred, labels):
    total = len(y_true)
    if total == 0:
        return 0.0
    score = 0.0
    for label in labels:
        support = sum(1 for y in y_true if y == label)
        if support == 0:
            continue
        tp = sum(1 for gold, pred in zip(y_true, y_pred) if gold == label and pred == label)
        fp = sum(1 for gold, pred in zip(y_true, y_pred) if gold != label and pred == label)
        fn = sum(1 for gold, pred in zip(y_true, y_pred) if gold == label and pred != label)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        score += (support / total) * f1
    return score


def resolve_model_class():
    import transformers

    for name in (
        "Qwen3VLForConditionalGeneration",
        "AutoModelForMultimodalLM",
        "AutoModelForImageTextToText",
    ):
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    raise ImportError(
        "Could not find Qwen3VLForConditionalGeneration or a compatible AutoModel class. "
        "Install a recent transformers build, e.g. pip install git+https://github.com/huggingface/transformers"
    )


def load_model_and_processor(args):
    import torch
    from transformers import AutoProcessor

    model_cls = resolve_model_class()
    dtype = torch.bfloat16 if args.dtype == "bf16" else "auto"

    def try_load(use_flash_attn):
        kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if use_flash_attn:
            kwargs["attn_implementation"] = "flash_attention_2"
        try:
            return model_cls.from_pretrained(args.model_path, dtype=dtype, **kwargs)
        except TypeError:
            return model_cls.from_pretrained(args.model_path, torch_dtype=dtype, **kwargs)

    try:
        model = try_load(args.flash_attn)
    except Exception as exc:
        if not args.flash_attn:
            raise
        print(
            f"[evaluate_qwen3vl] flash_attention_2 load failed, retrying without it: {exc}",
            file=sys.stderr,
            flush=True,
        )
        model = try_load(False)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    return model, processor


def build_messages(row, labels, fps, prompt_field):
    prompt = strict_prompt(row.get(prompt_field) or row.get("qwen_prompt") or row.get("text"), labels)
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": row["video_path"], "fps": fps},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def model_input_device(model):
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def generate_one(model, processor, row, labels, args):
    import torch

    messages = build_messages(row, labels, args.fps, args.prompt_field)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model_input_device(model))
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return output_text


def existing_sample_ids(path):
    if not path.exists():
        return set()
    ids = set()
    for row in read_jsonl(path):
        if row.get("sample_id"):
            ids.add(row["sample_id"])
    return ids


def run_inference(args):
    rows = list(read_jsonl(Path(args.manifest)))
    labels = load_labels(args.label_file, rows)
    shard_rows = [row for idx, row in enumerate(rows) if idx % args.world_size == args.rank]
    output_path = Path(args.output_dir) / f"{args.dataset}_{args.split}_shard{args.rank}.jsonl"
    done_ids = existing_sample_ids(output_path) if args.resume else set()

    print(
        json.dumps(
            {
                "event": "start",
                "dataset": args.dataset,
                "split": args.split,
                "rank": args.rank,
                "world_size": args.world_size,
                "rows_total": len(rows),
                "rows_shard": len(shard_rows),
                "rows_done": len(done_ids),
                "labels": labels,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    model = None
    processor = None
    for local_idx, row in enumerate(shard_rows, 1):
        sample_id = row.get("sample_id", f"rank{args.rank}_{local_idx}")
        if sample_id in done_ids:
            continue

        started = time.time()
        gold = str(row.get("emotion", "")).lower()
        video_path = row.get("video_path")
        if not video_path or not Path(video_path).exists():
            append_jsonl(
                output_path,
                {
                    "sample_id": sample_id,
                    "status": "skipped_missing_video",
                    "gold": gold,
                    "prediction": None,
                    "raw_output": "",
                    "video_path": video_path,
                },
            )
            continue

        if model is None or processor is None:
            model, processor = load_model_and_processor(args)

        try:
            raw_output = generate_one(model, processor, row, labels, args)
            prediction = extract_label(raw_output, labels)
            status = "ok" if prediction is not None else "ok_unparsed"
            error = None
        except Exception as exc:
            raw_output = ""
            prediction = None
            status = "failed"
            error = repr(exc)

        append_jsonl(
            output_path,
            {
                "sample_id": sample_id,
                "dataset": args.dataset,
                "split": args.split,
                "status": status,
                "gold": gold,
                "prediction": prediction,
                "raw_output": raw_output,
                "correct": prediction == gold,
                "video_path": video_path,
                "latency_sec": round(time.time() - started, 4),
                "error": error,
            },
        )
        if local_idx % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "rank": args.rank,
                        "processed_shard_rows": local_idx,
                        "total_shard_rows": len(shard_rows),
                    }
                ),
                flush=True,
            )


def aggregate(args):
    output_dir = Path(args.output_dir)
    pattern = f"{args.dataset}_{args.split}_shard*.jsonl"
    files = sorted(output_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No prediction shards matched {output_dir / pattern}")

    rows = []
    for file in files:
        rows.extend(read_jsonl(file))

    manifest_rows = list(read_jsonl(Path(args.manifest))) if args.manifest else []
    labels = load_labels(args.label_file, manifest_rows or rows)
    status_counts = Counter(row.get("status", "unknown") for row in rows)
    eval_rows = [row for row in rows if str(row.get("status", "")).startswith("ok")]
    y_true = [row["gold"] for row in eval_rows]
    y_pred = [row.get("prediction") or "__invalid__" for row in eval_rows]
    correct = sum(1 for gold, pred in zip(y_true, y_pred) if gold == pred)
    acc = correct / len(eval_rows) if eval_rows else 0.0
    wf1 = weighted_f1(y_true, y_pred, labels)
    confusion = Counter(f"{gold} -> {pred}" for gold, pred in zip(y_true, y_pred))

    metrics = {
        "dataset": args.dataset,
        "split": args.split,
        "prediction_files": [str(file) for file in files],
        "num_records": len(rows),
        "num_eval_records": len(eval_rows),
        "num_manifest_records": len(manifest_rows) if manifest_rows else None,
        "status_counts": dict(sorted(status_counts.items())),
        "labels": labels,
        "accuracy": acc,
        "weighted_f1": wf1,
        "correct": correct,
        "confusion": dict(sorted(confusion.items())),
    }
    metrics_path = output_dir / f"{args.dataset}_{args.split}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def default_manifest(project_root, dataset, split):
    if dataset == "meld":
        return project_root / f"datasets_video_text/data/meld/processed/{split}.jsonl"
    if dataset == "iemocap":
        return project_root / f"datasets_video_text/data/iemocap/processed_sentence/{split}.jsonl"
    raise ValueError(f"Unknown dataset: {dataset}")


def default_label_file(project_root, dataset):
    if dataset == "meld":
        return project_root / "datasets_video_text/data/meld/processed/labels.json"
    if dataset == "iemocap":
        return project_root / "datasets_video_text/data/iemocap/processed_sentence/labels.json"
    raise ValueError(f"Unknown dataset: {dataset}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["infer", "aggregate"], default="infer")
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--dataset", choices=["meld", "iemocap"], required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--manifest")
    parser.add_argument("--label-file")
    parser.add_argument("--output-dir")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--prompt-field", default="qwen_prompt")
    parser.add_argument("--dtype", choices=["bf16", "auto"], default="bf16")
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    project_root = Path(args.project_root)
    if args.manifest is None:
        args.manifest = str(default_manifest(project_root, args.dataset, args.split))
    if args.label_file is None:
        args.label_file = str(default_label_file(project_root, args.dataset))
    if args.output_dir is None:
        args.output_dir = str(project_root / "datasets_video_text/results/qwen3vl_8b")
    return args


def main():
    args = parse_args()
    if args.mode == "infer":
        run_inference(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"[evaluate_qwen3vl] ERROR: {exc}", file=sys.stderr)
        raise
