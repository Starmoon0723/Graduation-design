#!/usr/bin/env python3
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


SPLIT_FILES = {
    "train": ["train_sent_emo.csv"],
    "dev": ["dev_sent_emo.csv"],
    "test": ["test_sent_emo.csv"],
}

VIDEO_DIR_HINTS = {
    "train": ["train_splits", "train_splits_complete"],
    "dev": ["dev_splits_complete", "dev_splits"],
    "test": ["output_repeated_splits_test", "test_splits_complete", "test_splits"],
}


def find_file(root: Path, names):
    wanted = {name.lower() for name in names}
    for path in root.rglob("*"):
        if path.is_file() and path.name.lower() in wanted:
            return path
    found = sorted(str(path) for path in root.rglob("*.csv"))
    found_text = "\n".join(found[:30]) if found else "(no csv files found)"
    raise FileNotFoundError(
        f"Could not find any of {names} below {root}.\n"
        f"CSV files currently visible below raw-dir:\n{found_text}"
    )


def load_video_index(root: Path):
    index = {}
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        for path in root.rglob(ext):
            index[path.name.lower()] = path.resolve()
    return index


def resolve_video(video_index, dialogue_id, utterance_id):
    names = [
        f"dia{dialogue_id}_utt{utterance_id}.mp4",
        f"dia{dialogue_id}_utt{utterance_id}.avi",
        f"dia{dialogue_id}_utt{utterance_id}.mov",
        f"dia{dialogue_id}_utt{utterance_id}.mkv",
    ]
    for name in names:
        path = video_index.get(name.lower())
        if path:
            return str(path)
    return None


def make_prompt(context, speaker, text, labels):
    context_lines = []
    for turn in context[-8:]:
        context_lines.append(f"{turn['speaker']}: {turn['text']}")
    context_text = "\n".join(context_lines) if context_lines else "(none)"
    label_text = ", ".join(labels)
    return (
        "You are given a short video clip and its dialogue transcript. "
        "Predict the emotion of the target utterance.\n"
        f"Candidate emotions: {label_text}\n"
        f"Previous context:\n{context_text}\n"
        f"Target speaker: {speaker}\n"
        f"Target utterance: {text}\n"
        "Answer with exactly one candidate emotion."
    )


def normalize_row(row):
    return {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--context-window", type=int, default=8)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    video_index = load_video_index(raw_dir)
    all_rows = {}
    labels = set()

    for split, names in SPLIT_FILES.items():
        csv_path = find_file(raw_dir, names)
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = [normalize_row(row) for row in csv.DictReader(f)]
        all_rows[split] = rows
        labels.update(row["Emotion"].lower() for row in rows if row.get("Emotion"))

    labels_sorted = sorted(labels)
    summary = {"dataset": "meld", "splits": {}, "missing_videos": {}}

    for split, rows in all_rows.items():
        by_dialogue = defaultdict(list)
        for row in rows:
            by_dialogue[row["Dialogue_ID"]].append(row)

        for dialogue_rows in by_dialogue.values():
            dialogue_rows.sort(key=lambda r: int(r["Utterance_ID"]))

        out_path = output_dir / f"{split}.jsonl"
        emotion_counter = Counter()
        missing_videos = 0
        total = 0

        with out_path.open("w", encoding="utf-8") as out:
            for dialogue_id in sorted(by_dialogue, key=lambda x: int(x)):
                history = []
                for row in by_dialogue[dialogue_id]:
                    utterance_id = row["Utterance_ID"]
                    text = row["Utterance"]
                    speaker = row["Speaker"]
                    emotion = row["Emotion"].lower()
                    sentiment = row.get("Sentiment", "").lower() or None
                    context = history[-args.context_window :]
                    video_path = resolve_video(video_index, dialogue_id, utterance_id)
                    if video_path is None:
                        missing_videos += 1
                    record = {
                        "dataset": "meld",
                        "split": split,
                        "sample_id": f"meld_{split}_dia{dialogue_id}_utt{utterance_id}",
                        "dialogue_id": str(dialogue_id),
                        "utterance_id": str(utterance_id),
                        "speaker": speaker,
                        "text": text,
                        "emotion": emotion,
                        "sentiment": sentiment,
                        "video_path": video_path,
                        "context": context,
                        "qwen_prompt": make_prompt(context, speaker, text, labels_sorted),
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    history.append({"speaker": speaker, "text": text, "emotion": emotion})
                    emotion_counter[emotion] += 1
                    total += 1

        summary["splits"][split] = {
            "samples": total,
            "emotion_counts": dict(sorted(emotion_counter.items())),
        }
        summary["missing_videos"][split] = missing_videos

    (output_dir / "labels.json").write_text(
        json.dumps({"emotion": labels_sorted}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
