#!/usr/bin/env python3
import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path

from config_utils import load_config, resolve_project_path
from generate_reasoning_with_vllm import (
    conversation_text,
    count_fusion_flags,
    dataset_cfg,
    extract_reason,
    has_instruction_leak,
    load_labels,
    normalize_label,
    parse_fusion_response,
    read_jsonl,
    speaker_display,
)


THINK_WORD_RANGE = (80, 320)
DIALOGUE_VISUAL_LEAK_TERMS = (
    "video",
    "visual",
    "face",
    "facial",
    "gesture",
    "gaze",
    "posture",
    "body language",
    "audio",
    "voice",
    "scene",
)
VISUAL_FINAL_LEAK_PATTERNS = (
    r"\bfinal_answer\b",
    r"<answer>",
    r"\bfinal\s+(emotion|label|answer)\b",
    r"\bcandidate emotion labels\b",
    r"\bcandidate labels\b",
)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def word_count(text):
    return len(re.findall(r"[A-Za-z0-9']+", text or ""))


def contains_any(text, terms):
    lowered = (text or "").lower()
    for term in terms:
        term = term.lower()
        if " " in term:
            if term in lowered:
                return True
        elif re.search(rf"\b{re.escape(term)}\b", lowered):
            return True
    return False


def has_visual_final_leak(text, labels):
    lowered = (text or "").lower()
    if any(re.search(pattern, lowered) for pattern in VISUAL_FINAL_LEAK_PATTERNS):
        return True
    label_list = ", ".join(labels).lower()
    return label_list in lowered


def load_rows_by_id(patterns, expected_step, reason_key=None):
    by_id = {}
    for pattern in patterns:
        for file in sorted(glob.glob(str(pattern))):
            for row in read_jsonl(file):
                if row.get("status") != "ok":
                    continue
                if row.get("step") != expected_step:
                    continue
                sample_id = row.get("sample_id")
                if not sample_id:
                    continue
                if reason_key and not row.get(reason_key):
                    continue
                by_id[sample_id] = row
    return by_id


def default_patterns(cfg, dataset, split, step):
    key_map = {
        "visual": "step1_visual_reason",
        "dialogue": "step2_dialogue_reason",
        "fusion": "step3_fusion_reason",
    }
    root = resolve_project_path(cfg, cfg["output"]["root"]) / cfg["output"][key_map[step]]
    return [root / dataset / f"{split}_shard*.jsonl"]


def output_paths(cfg, dataset, split):
    root = resolve_project_path(cfg, cfg["output"]["root"])
    output_cfg = cfg["output"]
    return {
        "sft": root / output_cfg.get("final_sft", "final_sft") / dataset / f"{split}.jsonl",
        "rl": root / output_cfg.get("final_rl", "final_rl") / dataset / f"{split}.jsonl",
        "preference": root / output_cfg.get("final_preferences", "final_preferences") / dataset / f"{split}.jsonl",
        "summary": root / output_cfg.get("final_sft", "final_sft") / dataset / f"{split}_summary.json",
    }


def default_manifest(cfg, dataset, split):
    item = dataset_cfg(cfg, dataset)
    return resolve_project_path(cfg, item["manifest_dir"]) / f"{split}.jsonl"


def default_label_file(cfg, dataset):
    item = dataset_cfg(cfg, dataset)
    return resolve_project_path(cfg, item["label_file"])


def build_student_user_prompt(row, labels):
    label_text = ", ".join(labels)
    return (
        "<video>\n"
        "You are an expert in multimodal emotion recognition in conversation.\n\n"
        "You will be given the video clip of the current utterance and the dialogue context. "
        "Analyze the current speaker's emotion using both the video and the dialogue context.\n\n"
        "### Dialogue Context\n"
        f"{conversation_text(row)}\n\n"
        "### Current Speaker\n"
        f"{speaker_display(row)}\n\n"
        "### Current Utterance\n"
        f"\"{str(row.get('text', '')).strip()}\"\n\n"
        "### Candidate Emotion Labels\n"
        f"{label_text}\n\n"
        "### Task\n"
        "First reason about the current speaker's emotion using the visual evidence from "
        "the video and the textual evidence from the dialogue context. Then output exactly "
        "one final emotion label.\n\n"
        "Output format:\n"
        "<think>\n"
        "...\n"
        "</think>\n"
        "<answer>\n"
        "one lowercase label from the candidate labels\n"
        "</answer>"
    )


def fallback_reason_from_row(row, field_name, key_name):
    reason = row.get(key_name)
    if reason:
        return reason
    reason, status = extract_reason(row.get("teacher_output", ""), field_name, strict_schema=True)
    return reason if status == "ok" else None


def validate_sample(row, labels, visual_row, dialogue_row, fusion_row, args):
    failures = []
    gold = normalize_label(row.get(args.gold_field, ""))
    video_path = row.get("video_path")
    visual_reason = fallback_reason_from_row(visual_row or {}, "VISUAL_REASON", "visual_reason")
    dialogue_reason = fallback_reason_from_row(dialogue_row or {}, "DIALOGUE_REASON", "dialogue_reason")
    raw_fusion_output = (fusion_row or {}).get("teacher_output", "")
    fusion_response = (fusion_row or {}).get("fusion_response")
    if fusion_response:
        fusion_reason = (fusion_row or {}).get("fusion_reason")
        final_answer = normalize_label((fusion_row or {}).get("final_answer"))
        parsed_response = fusion_response
        fusion_status = "ok" if fusion_reason and final_answer == gold and final_answer in labels else "parse_failed"
        parse_error = None if fusion_status == "ok" else "invalid_cached_fusion_response"
    else:
        fusion_reason, final_answer, parsed_response, fusion_status, parse_error = parse_fusion_response(
            raw_fusion_output, gold, labels
        )

    if not visual_row:
        failures.append("missing_visual_reason")
    if not dialogue_row:
        failures.append("missing_dialogue_reason")
    if not fusion_row:
        failures.append("missing_fusion_response")
    if not video_path:
        failures.append("missing_video")
    elif not args.skip_video_exists_check and not Path(video_path).exists():
        failures.append("missing_video")
    if gold not in labels:
        failures.append("invalid_gold_label")
    if not visual_reason:
        failures.append("missing_visual_reason_text")
    if not dialogue_reason:
        failures.append("missing_dialogue_reason_text")
    elif contains_any(dialogue_reason, DIALOGUE_VISUAL_LEAK_TERMS):
        failures.append("dialogue_visual_leak")
    if visual_reason and has_visual_final_leak(visual_reason, labels):
        failures.append("visual_final_label_leak")
    if fusion_status != "ok":
        failures.append("fusion_parse_failed")
        if parse_error:
            failures.append(f"fusion_parse_failed_{parse_error}")
    if fusion_reason:
        think_words = word_count(fusion_reason)
        if think_words < THINK_WORD_RANGE[0]:
            failures.append("fusion_think_too_short")
        elif think_words > THINK_WORD_RANGE[1]:
            failures.append("fusion_think_too_long")
        if has_instruction_leak(fusion_reason):
            failures.append("fusion_instruction_leak")

    return {
        "failures": failures,
        "gold": gold,
        "visual_reason": visual_reason,
        "dialogue_reason": dialogue_reason,
        "fusion_response": parsed_response,
        "fusion_reason": fusion_reason,
        "final_answer": final_answer,
        "fusion_flags": count_fusion_flags(fusion_reason or ""),
    }


def make_sft_row(row, labels, quality):
    query = build_student_user_prompt(row, labels)
    response = quality["fusion_response"].strip()
    return {
        "sample_id": row.get("sample_id"),
        "dataset": row.get("dataset"),
        "split": row.get("split"),
        "task_type": "multimodal_reasoning_sft",
        "messages": [
            {"role": "user", "content": query},
            {"role": "assistant", "content": response},
        ],
        "query": query,
        "response": response,
        "videos": [row.get("video_path")],
        "gold": quality["gold"],
        "candidate_labels": labels,
        "visual_reason": quality["visual_reason"],
        "dialogue_reason": quality["dialogue_reason"],
        "fusion_response": response,
    }


def make_rl_row(row, labels, quality):
    return {
        "sample_id": row.get("sample_id"),
        "dataset": row.get("dataset"),
        "split": row.get("split"),
        "task_type": "multimodal_reasoning_grpo",
        "prompt": build_student_user_prompt(row, labels),
        "videos": [row.get("video_path")],
        "gold": quality["gold"],
        "candidate_labels": labels,
        "reference_visual_reason": quality["visual_reason"],
        "reference_dialogue_reason": quality["dialogue_reason"],
        "reference_fusion_response": quality["fusion_response"].strip(),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", choices=["meld", "iemocap"], required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--manifest")
    parser.add_argument("--label-file")
    parser.add_argument("--visual-pattern", action="append")
    parser.add_argument("--dialogue-pattern", action="append")
    parser.add_argument("--fusion-pattern", action="append")
    parser.add_argument("--skip-video-exists-check", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    item = dataset_cfg(cfg, args.dataset)
    args.gold_field = item.get("gold_field", "emotion_prompt")
    args.manifest = args.manifest or str(default_manifest(cfg, args.dataset, args.split))
    args.label_file = args.label_file or str(default_label_file(cfg, args.dataset))
    args.visual_pattern = args.visual_pattern or [str(p) for p in default_patterns(cfg, args.dataset, args.split, "visual")]
    args.dialogue_pattern = args.dialogue_pattern or [str(p) for p in default_patterns(cfg, args.dataset, args.split, "dialogue")]
    args.fusion_pattern = args.fusion_pattern or [str(p) for p in default_patterns(cfg, args.dataset, args.split, "fusion")]
    args.output_paths = output_paths(cfg, args.dataset, args.split)
    return args


def main():
    args = parse_args()
    labels = load_labels(args.label_file)
    source_rows = list(read_jsonl(args.manifest))
    visual_by_id = load_rows_by_id(args.visual_pattern, "visual", "visual_reason")
    dialogue_by_id = load_rows_by_id(args.dialogue_pattern, "dialogue", "dialogue_reason")
    fusion_by_id = load_rows_by_id(args.fusion_pattern, "fusion")

    sft_rows = []
    rl_rows = []
    pref_rows = []
    stats = Counter(total_manifest_samples=len(source_rows))
    failure_examples = []

    for row in source_rows:
        sid = row.get("sample_id")
        visual_row = visual_by_id.get(sid)
        dialogue_row = dialogue_by_id.get(sid)
        fusion_row = fusion_by_id.get(sid)
        if visual_row:
            stats["visual_loaded_ok"] += 1
        if dialogue_row:
            stats["dialogue_loaded_ok"] += 1
        if fusion_row:
            stats["fusion_loaded_ok"] += 1
            stats["fusion_generated_ok"] += 1

        quality = validate_sample(row, labels, visual_row, dialogue_row, fusion_row, args)
        if quality["failures"]:
            for failure in quality["failures"]:
                stats[failure] += 1
                if failure in ("fusion_think_too_short", "fusion_think_too_long"):
                    stats["filtered_by_reason_length"] += 1
                if failure in ("fusion_instruction_leak",):
                    stats["filtered_by_instruction_leak"] += 1
            if len(failure_examples) < 30:
                failure_examples.append({"sample_id": sid, "failures": quality["failures"]})
            continue

        stats["passed_quality_filter"] += 1
        for flag_name, enabled in quality["fusion_flags"].items():
            if enabled:
                stats[flag_name] += 1
        sft_rows.append(make_sft_row(row, labels, quality))
        rl_rows.append(make_rl_row(row, labels, quality))

    counts = {
        "final_sft_samples": write_jsonl(args.output_paths["sft"], sft_rows),
        "final_rl_samples": write_jsonl(args.output_paths["rl"], rl_rows),
        "final_preference_samples": write_jsonl(args.output_paths["preference"], pref_rows),
    }
    stats.update(counts)
    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "manifest": args.manifest,
        "labels": labels,
        "gold_field": args.gold_field,
        "visual_patterns": args.visual_pattern,
        "dialogue_patterns": args.dialogue_pattern,
        "fusion_patterns": args.fusion_pattern,
        "quality_rules": {
            "think_words": list(THINK_WORD_RANGE),
            "dialogue_visual_leak_terms": list(DIALOGUE_VISUAL_LEAK_TERMS),
            "visual_final_leak_patterns": list(VISUAL_FINAL_LEAK_PATTERNS),
        },
        "outputs": {name: str(path) for name, path in args.output_paths.items()},
        "stats": dict(sorted(stats.items())),
        "failure_examples": failure_examples,
    }
    args.output_paths["summary"].parent.mkdir(parents=True, exist_ok=True)
    args.output_paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
