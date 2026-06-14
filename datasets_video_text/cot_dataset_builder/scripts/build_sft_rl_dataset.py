#!/usr/bin/env python3
import argparse
import glob
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from config_utils import load_config, resolve_project_path
from generate_reasoning_with_vllm import (
    conversation_text,
    dataset_cfg,
    extract_reason,
    load_labels,
    normalize_label,
    read_jsonl,
    speaker_name,
)


VISUAL_WORD_RANGE = (30, 220)
DIALOGUE_WORD_RANGE = (50, 260)
VISUAL_LEAK_TERMS = (
    "face",
    "facial",
    "facial expression",
    "gesture",
    "gestures",
    "gaze",
    "video",
    "frame",
    "frames",
    "visible",
    "visual",
    "body language",
    "posture",
    "eye contact",
    "smile",
    "frown",
)
FINAL_LABEL_PATTERNS = (
    r"\bfinal\s+(emotion|label|answer)\s+is\b",
    r"\bthe\s+final\s+(emotion|label|answer)\b",
    r"\bfinal_answer\b",
    r"<answer>",
    r"\bthe\s+emotion\s+is\s+best\s+identified\s+as\b",
    r"\bthe\s+speaker'?s\s+emotion\s+is\b",
)

INTEGRATION_TEMPLATES = [
    'Taken together, the visible behavior and the conversational context support the label "{gold}".',
    'The dialogue provides the main emotional direction, while the video evidence offers additional grounding for "{gold}".',
    'The visual cues and textual context are consistent with the target emotion "{gold}".',
    'Although the visual evidence may be subtle, it can be considered together with the dialogue context to support "{gold}".',
    'Both sources of evidence point toward "{gold}" when the utterance is interpreted in context.',
    'The observable behavior and the wording of the utterance jointly make "{gold}" the most appropriate label.',
    'The video evidence grounds the speaker state, and the dialogue evidence explains why "{gold}" fits this turn.',
    'Considering the visual evidence alongside the conversational meaning, "{gold}" is the best supported emotion.',
]


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


def has_final_label_leak(text):
    lowered = (text or "").lower()
    return any(re.search(pattern, lowered) for pattern in FINAL_LABEL_PATTERNS)


def choose_template(sample_id):
    digest = hashlib.md5(str(sample_id).encode("utf-8")).hexdigest()
    return INTEGRATION_TEMPLATES[int(digest[:8], 16) % len(INTEGRATION_TEMPLATES)]


def choose_wrong_label(sample_id, labels, gold):
    candidates = [label for label in labels if label != gold]
    if not candidates:
        return None
    digest = hashlib.md5((str(sample_id) + gold).encode("utf-8")).hexdigest()
    return candidates[int(digest[:8], 16) % len(candidates)]


def source_key(row):
    return row.get("sample_id")


def reason_from_teacher(row, step):
    field = "VISUAL_REASON" if step == "visual" else "DIALOGUE_REASON"
    key = "visual_reason" if step == "visual" else "dialogue_reason"
    reason = (row.get(key) or "").strip()
    if reason:
        return reason
    reason, _ = extract_reason(row.get("teacher_output", ""), field)
    return reason.strip()


def load_teacher_outputs(patterns, step):
    by_id = {}
    reason_key = "visual_reason" if step == "visual" else "dialogue_reason"
    for pattern in patterns:
        for file in sorted(glob.glob(str(pattern))):
            for row in read_jsonl(file):
                sid = source_key(row)
                if not sid:
                    continue
                if row.get("status") != "ok":
                    continue
                if reason_key not in row:
                    continue
                copied = dict(row)
                copied[f"{step}_reason"] = reason_from_teacher(row, step)
                by_id[sid] = copied
    return by_id


def default_patterns(cfg, dataset, split, step):
    step_key = "step1_visual_reason" if step == "visual" else "step2_dialogue_reason"
    root = resolve_project_path(cfg, cfg["output"]["root"]) / cfg["output"][step_key]
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
        "Analyze the current speaker's emotion using both the visual evidence from the video "
        "and the textual evidence from the conversation.\n\n"
        "### Dialogue Context\n"
        f"{conversation_text(row)}\n\n"
        "### Current Speaker\n"
        f"{speaker_name(row)}\n\n"
        "### Current Utterance\n"
        f"\"{str(row.get('text', '')).strip()}\"\n\n"
        "### Candidate Emotion Labels\n"
        f"{label_text}\n\n"
        "### Task\n"
        "First reason about the speaker's emotion using visual evidence and dialogue evidence. "
        "Then output exactly one final emotion label.\n\n"
        "Output format:\n"
        "<think>\n"
        "...\n"
        "</think>\n"
        "<answer>\n"
        "one lowercase label from the candidate labels\n"
        "</answer>"
    )


def build_sft_assistant_response(visual_reason, dialogue_reason, gold, sample_id):
    integration = choose_template(sample_id).format(gold=gold)
    return (
        "<think>\n"
        f"Visual evidence: {visual_reason.strip()}\n\n"
        f"Dialogue evidence: {dialogue_reason.strip()}\n\n"
        f"{integration}\n"
        "</think>\n"
        "<answer>\n"
        f"{gold}\n"
        "</answer>"
    )


def check_quality(row, labels, visual_reason, dialogue_reason, args):
    reasons = []
    gold = normalize_label(row.get(args.gold_field, ""))
    video_path = row.get("video_path")
    visual_words = word_count(visual_reason)
    dialogue_words = word_count(dialogue_reason)

    if not visual_reason:
        reasons.append("missing_visual_reason")
    elif visual_words < VISUAL_WORD_RANGE[0]:
        reasons.append("visual_too_short")
    elif visual_words > VISUAL_WORD_RANGE[1]:
        reasons.append("visual_too_long")

    if not dialogue_reason:
        reasons.append("missing_dialogue_reason")
    elif dialogue_words < DIALOGUE_WORD_RANGE[0]:
        reasons.append("dialogue_too_short")
    elif dialogue_words > DIALOGUE_WORD_RANGE[1]:
        reasons.append("dialogue_too_long")

    if dialogue_reason and contains_any(dialogue_reason, VISUAL_LEAK_TERMS):
        reasons.append("dialogue_visual_leak")
    if visual_reason and has_final_label_leak(visual_reason):
        reasons.append("visual_final_label_leak")
    if gold not in labels:
        reasons.append("invalid_gold_label")
    if not video_path:
        reasons.append("missing_video")
    elif not args.skip_video_exists_check and not Path(video_path).exists():
        reasons.append("missing_video")

    return reasons


def make_sft_row(row, labels, visual_reason, dialogue_reason, gold):
    query = build_student_user_prompt(row, labels)
    response = build_sft_assistant_response(visual_reason, dialogue_reason, gold, row.get("sample_id"))
    video_path = row.get("video_path")
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
        "videos": [video_path],
        "gold": gold,
        "candidate_labels": labels,
        "visual_reason": visual_reason,
        "dialogue_reason": dialogue_reason,
    }


def make_rl_row(row, labels, visual_reason, dialogue_reason, gold):
    return {
        "sample_id": row.get("sample_id"),
        "dataset": row.get("dataset"),
        "split": row.get("split"),
        "task_type": "multimodal_reasoning_grpo",
        "prompt": build_student_user_prompt(row, labels),
        "videos": [row.get("video_path")],
        "gold": gold,
        "candidate_labels": labels,
        "reference_visual_reason": visual_reason,
        "reference_dialogue_reason": dialogue_reason,
    }


def make_weak_preference_row(row, labels, visual_reason, dialogue_reason, gold):
    wrong_label = choose_wrong_label(row.get("sample_id"), labels, gold)
    if not wrong_label:
        return None
    return {
        "sample_id": row.get("sample_id"),
        "dataset": row.get("dataset"),
        "split": row.get("split"),
        "task_type": "multimodal_reasoning_preference",
        "preference_type": "weak_answer_only_wrong_label",
        "prompt": build_student_user_prompt(row, labels),
        "videos": [row.get("video_path")],
        "chosen": build_sft_assistant_response(visual_reason, dialogue_reason, gold, row.get("sample_id")),
        "rejected": f"<answer>\n{wrong_label}\n</answer>",
        "gold": gold,
        "rejected_label": wrong_label,
        "candidate_labels": labels,
        "visual_reason": visual_reason,
        "dialogue_reason": dialogue_reason,
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
    parser.add_argument("--build-weak-preferences", action="store_true")
    parser.add_argument("--skip-video-exists-check", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    item = dataset_cfg(cfg, args.dataset)
    args.gold_field = item.get("gold_field", "emotion_prompt")
    args.manifest = args.manifest or str(default_manifest(cfg, args.dataset, args.split))
    args.label_file = args.label_file or str(default_label_file(cfg, args.dataset))
    args.visual_pattern = args.visual_pattern or [str(p) for p in default_patterns(cfg, args.dataset, args.split, "visual")]
    args.dialogue_pattern = args.dialogue_pattern or [str(p) for p in default_patterns(cfg, args.dataset, args.split, "dialogue")]
    args.output_paths = output_paths(cfg, args.dataset, args.split)
    return args


def main():
    args = parse_args()
    labels = load_labels(args.label_file)
    source_rows = list(read_jsonl(args.manifest))
    visual_by_id = load_teacher_outputs(args.visual_pattern, "visual")
    dialogue_by_id = load_teacher_outputs(args.dialogue_pattern, "dialogue")

    sft_rows = []
    rl_rows = []
    pref_rows = []
    stats = Counter(total_samples=len(source_rows))
    failure_examples = []

    for row in source_rows:
        sid = row.get("sample_id")
        visual_row = visual_by_id.get(sid)
        dialogue_row = dialogue_by_id.get(sid)
        if visual_row:
            stats["visual_teacher_rows"] += 1
        else:
            stats["missing_visual_teacher"] += 1
        if dialogue_row:
            stats["dialogue_teacher_rows"] += 1
        else:
            stats["missing_dialogue_teacher"] += 1

        visual_reason = (visual_row or {}).get("visual_reason", "")
        dialogue_reason = (dialogue_row or {}).get("dialogue_reason", "")
        if visual_reason:
            stats["visual_reason_generated"] += 1
        if dialogue_reason:
            stats["dialogue_reason_generated"] += 1
        failures = check_quality(row, labels, visual_reason, dialogue_reason, args)
        if failures:
            for failure in failures:
                stats[failure] += 1
            if len(failure_examples) < 30:
                failure_examples.append({"sample_id": sid, "failures": failures})
            continue

        gold = normalize_label(row.get(args.gold_field, ""))
        stats["passed_quality_filter"] += 1
        sft_rows.append(make_sft_row(row, labels, visual_reason, dialogue_reason, gold))
        rl_rows.append(make_rl_row(row, labels, visual_reason, dialogue_reason, gold))
        if args.build_weak_preferences:
            pref = make_weak_preference_row(row, labels, visual_reason, dialogue_reason, gold)
            if pref:
                pref_rows.append(pref)

    counts = {
        "final_sft_samples": write_jsonl(args.output_paths["sft"], sft_rows),
        "final_rl_samples": write_jsonl(args.output_paths["rl"], rl_rows),
        "preference_samples": write_jsonl(args.output_paths["preference"], pref_rows),
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
        "quality_rules": {
            "visual_words": list(VISUAL_WORD_RANGE),
            "dialogue_words": list(DIALOGUE_WORD_RANGE),
            "dialogue_visual_leak_terms": list(VISUAL_LEAK_TERMS),
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
