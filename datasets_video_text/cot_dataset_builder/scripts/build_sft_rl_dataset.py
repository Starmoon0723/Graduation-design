#!/usr/bin/env python3
import argparse
import glob
import json
from collections import Counter
from pathlib import Path

from config_utils import load_config, resolve_project_path
from generate_reasoning_with_vllm import build_prompt, dataset_cfg, load_labels, read_jsonl


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def source_key(row):
    return row.get("sample_id")


def load_teacher_outputs(patterns):
    by_id = {}
    for pattern in patterns:
        for file in sorted(glob.glob(str(pattern))):
            for row in read_jsonl(file):
                sid = source_key(row)
                if not sid:
                    continue
                by_id[sid] = row
    return by_id


def default_patterns(cfg, dataset, split, step):
    step_key = "step1_visual_reason" if step == "visual" else "step2_dialogue_reason"
    root = resolve_project_path(cfg, cfg["output"]["root"]) / cfg["output"][step_key]
    return [root / dataset / f"{split}_shard*.jsonl"]


def default_out_dir(cfg, dataset, split):
    root = resolve_project_path(cfg, cfg["output"]["root"]) / cfg["output"]["step3_sft_rl"]
    return root / dataset / split


def build_user_content(row, labels, prompt_field, step):
    prompt = build_prompt(row, labels, prompt_field, step)
    content = []
    if step == "visual":
        content.append({"type": "video_url", "video_url": {"url": "file://" + str(Path(row["video_path"]).resolve())}})
    content.append({"type": "text", "text": prompt})
    return content


def make_sft_row(source_row, teacher_row, labels, prompt_field, step):
    return {
        "sample_id": source_row["sample_id"],
        "dataset": source_row.get("dataset"),
        "split": source_row.get("split"),
        "task": f"{step}_emotion_reasoning_sft",
        "messages": [
            {
                "role": "user",
                "content": build_user_content(source_row, labels, prompt_field, step),
            },
            {
                "role": "assistant",
                "content": teacher_row["teacher_output"].strip(),
            },
        ],
        "metadata": {
            "gold": teacher_row.get("gold"),
            "teacher_prediction": teacher_row.get("prediction"),
            "teacher_correct": teacher_row.get("correct"),
            "video_path": source_row.get("video_path"),
            "dialogue_id": source_row.get("dialogue_id"),
            "utterance_id": source_row.get("utterance_id"),
            "speaker": source_row.get("speaker"),
        },
    }


def make_rl_row(source_row, teacher_row, labels, prompt_field, step):
    prompt_content = build_user_content(source_row, labels, prompt_field, step)
    gold = teacher_row.get("gold")
    pred = teacher_row.get("prediction")
    reward = 1.0 if pred == gold else 0.0
    return {
        "sample_id": source_row["sample_id"],
        "dataset": source_row.get("dataset"),
        "split": source_row.get("split"),
        "task": f"{step}_emotion_reasoning_reward",
        "prompt": prompt_content,
        "response": teacher_row.get("teacher_output", "").strip(),
        "reward": reward,
        "reward_model": "exact_label_match",
        "gold": gold,
        "prediction": pred,
        "labels": labels,
        "metadata": {
            "status": teacher_row.get("status"),
            "video_path": source_row.get("video_path"),
            "dialogue_id": source_row.get("dialogue_id"),
            "utterance_id": source_row.get("utterance_id"),
        },
    }


def make_preference_row(source_row, teacher_row, labels, prompt_field, step):
    gold = teacher_row.get("gold")
    pred = teacher_row.get("prediction")
    if not gold or pred == gold:
        return None
    rejected = teacher_row.get("teacher_output", "").strip()
    chosen = (
        "OBSERVATION: The evidence should be reconsidered using the dialogue and available visual cues.\n"
        "CONTEXT_REASON: The target utterance should be interpreted in its local conversational context.\n"
        f"{'VISUAL_REASON' if step == 'visual' else 'DIALOGUE_REASON'}: confidence=medium; the previous response selected a label inconsistent with the reference annotation.\n"
        f"FINAL_REASON: The corrected emotion label is {gold}.\n"
        f"FINAL_ANSWER: {gold}"
    )
    return {
        "sample_id": source_row["sample_id"],
        "dataset": source_row.get("dataset"),
        "split": source_row.get("split"),
        "task": f"{step}_emotion_reasoning_preference",
        "prompt": build_user_content(source_row, labels, prompt_field, step),
        "chosen": chosen,
        "rejected": rejected,
        "gold": gold,
        "rejected_prediction": pred,
        "metadata": {
            "video_path": source_row.get("video_path"),
            "dialogue_id": source_row.get("dialogue_id"),
            "utterance_id": source_row.get("utterance_id"),
        },
    }


def default_manifest(cfg, dataset, split):
    item = dataset_cfg(cfg, dataset)
    return resolve_project_path(cfg, item["manifest_dir"]) / f"{split}.jsonl"


def default_label_file(cfg, dataset):
    item = dataset_cfg(cfg, dataset)
    return resolve_project_path(cfg, item["label_file"])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--step", choices=["visual", "dialogue"], default="visual")
    parser.add_argument("--dataset", choices=["meld", "iemocap"], required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--manifest")
    parser.add_argument("--label-file")
    parser.add_argument("--teacher-pattern", action="append")
    parser.add_argument("--output-dir")
    parser.add_argument("--include-incorrect-sft", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    item = dataset_cfg(cfg, args.dataset)
    args.prompt_field = item.get("prompt_field", "qwen_prompt_new")
    args.manifest = args.manifest or str(default_manifest(cfg, args.dataset, args.split))
    args.label_file = args.label_file or str(default_label_file(cfg, args.dataset))
    args.teacher_pattern = args.teacher_pattern or [str(p) for p in default_patterns(cfg, args.dataset, args.split, args.step)]
    args.output_dir = args.output_dir or str(default_out_dir(cfg, args.dataset, args.split))
    return args


def main():
    args = parse_args()
    labels = load_labels(args.label_file)
    source_rows = list(read_jsonl(args.manifest))
    teacher_by_id = load_teacher_outputs(args.teacher_pattern)

    sft_rows = []
    rl_rows = []
    pref_rows = []
    status_counter = Counter()
    for row in source_rows:
        sid = row.get("sample_id")
        teacher = teacher_by_id.get(sid)
        if not teacher:
            status_counter["missing_teacher"] += 1
            continue
        status_counter[teacher.get("status", "unknown")] += 1
        if not str(teacher.get("status", "")).startswith("ok"):
            continue
        rl_rows.append(make_rl_row(row, teacher, labels, args.prompt_field, args.step))
        pref = make_preference_row(row, teacher, labels, args.prompt_field, args.step)
        if pref:
            pref_rows.append(pref)
        if args.include_incorrect_sft or teacher.get("correct"):
            sft_rows.append(make_sft_row(row, teacher, labels, args.prompt_field, args.step))

    out_dir = Path(args.output_dir)
    counts = {
        "sft": write_jsonl(out_dir / f"{args.step}_sft.jsonl", sft_rows),
        "rl": write_jsonl(out_dir / f"{args.step}_rl_rewards.jsonl", rl_rows),
        "preference": write_jsonl(out_dir / f"{args.step}_preferences.jsonl", pref_rows),
    }
    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "step": args.step,
        "manifest": args.manifest,
        "teacher_patterns": args.teacher_pattern,
        "labels": labels,
        "source_rows": len(source_rows),
        "teacher_rows": len(teacher_by_id),
        "status_counts": dict(sorted(status_counter.items())),
        "outputs": {
            "sft": str(out_dir / f"{args.step}_sft.jsonl"),
            "rl": str(out_dir / f"{args.step}_rl_rewards.jsonl"),
            "preference": str(out_dir / f"{args.step}_preferences.jsonl"),
        },
        "counts": counts,
    }
    (out_dir / f"{args.step}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
