#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_PROJECT_ROOT = Path(
    "/XYFS01/HDD_POOL/hitsz_mszhang/hitsz_mszhang_1/MRC/MRC/MRC_project/others/AAA/vlm/Graduation-design"
)


LABEL_MAPS = {
    "meld": {
        "neutral": "neutral",
        "surprise": "surprise",
        "fear": "fear",
        "sadness": "sad",
        "joy": "joyful",
        "disgust": "disgust",
        "anger": "angry",
    },
    "iemocap": {
        "happiness": "happy",
        "sadness": "sad",
        "neutral": "neutral",
        "anger": "angry",
        "excitement": "excited",
        "frustration": "frustrated",
    },
}

LABEL_ORDERS = {
    "meld": ["neutral", "surprise", "fear", "sad", "joyful", "disgust", "angry"],
    "iemocap": ["happy", "sad", "neutral", "angry", "excited", "frustrated"],
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


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sort_key(row):
    if "start" in row:
        return float(row["start"])
    try:
        return int(row["utterance_id"])
    except Exception:
        return str(row.get("utterance_id", ""))


def speaker_alias(speaker, speaker_map):
    speaker_key = str(speaker)
    if speaker_key not in speaker_map:
        speaker_map[speaker_key] = f"Speaker_{len(speaker_map)}"
    return speaker_map[speaker_key]


def escape_text(text):
    return str(text).replace("\n", " ").replace("\r", " ").strip()


def make_new_prompt(turns, target_idx, labels):
    context = []
    for idx, turn in enumerate(turns[: target_idx + 1]):
        context.append(f'{turn["speaker_alias"]}:"{escape_text(turn["text"])}"')
    target = turns[target_idx]
    label_text = ", ".join(labels)
    context_text = "\t ".join(context)
    return (
        "Now you are expert of sentiment and emotional analysis. "
        "The following conversation noted between '### ###' involves several speakers. "
        f"### \t {context_text} ### "
        f'Please select the emotional label of < {target["speaker_alias"]}:"{escape_text(target["text"])}"> '
        f"from <{label_text}>:"
    )


def convert_split(dataset, input_path, output_path, drop_unsupported=True):
    rows = list(read_jsonl(input_path))
    by_dialogue = defaultdict(list)
    for row in rows:
        by_dialogue[str(row["dialogue_id"])].append(row)

    labels = LABEL_ORDERS[dataset]
    label_map = LABEL_MAPS[dataset]
    converted = []
    skipped = []
    emotion_counter = Counter()

    for dialogue_id in sorted(by_dialogue):
        dialogue_rows = sorted(by_dialogue[dialogue_id], key=sort_key)
        speaker_map = {}
        turns = []
        row_to_turn_idx = {}
        for row in dialogue_rows:
            alias = speaker_alias(row.get("speaker"), speaker_map)
            turn_idx = len(turns)
            row_to_turn_idx[row.get("sample_id")] = turn_idx
            turns.append(
                {
                    "speaker_alias": alias,
                    "text": row.get("text", ""),
                }
            )

        for row in dialogue_rows:
            original_emotion = str(row.get("emotion", "")).lower()
            prompt_emotion = label_map.get(original_emotion)
            if prompt_emotion is None:
                skipped.append(
                    {
                        "sample_id": row.get("sample_id"),
                        "emotion": original_emotion,
                        "reason": "unsupported_by_new_prompt_label_space",
                    }
                )
                if drop_unsupported:
                    continue
                prompt_emotion = original_emotion

            new_row = dict(row)
            target_idx = row_to_turn_idx[row.get("sample_id")]
            new_row["speaker_original"] = row.get("speaker")
            new_row["speaker"] = turns[target_idx]["speaker_alias"]
            new_row["speaker_map"] = speaker_map
            new_row["emotion_original"] = original_emotion
            new_row["emotion_prompt"] = prompt_emotion
            new_row["qwen_prompt_new"] = make_new_prompt(turns, target_idx, labels)
            converted.append(new_row)
            emotion_counter[prompt_emotion] += 1

    write_jsonl(output_path, converted)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "rows_in": len(rows),
        "rows_out": len(converted),
        "dropped": len(rows) - len(converted),
        "emotion_counts": dict(sorted(emotion_counter.items())),
        "skipped_examples": skipped[:20],
    }


def dataset_paths(project_root, dataset):
    if dataset == "meld":
        return (
            project_root / "datasets_video_text/data/meld/processed",
            project_root / "datasets_video_text/data/meld/processed_new_prompt",
        )
    if dataset == "iemocap":
        return (
            project_root / "datasets_video_text/data/iemocap/processed_sentence",
            project_root / "datasets_video_text/data/iemocap/processed_sentence_new_prompt",
        )
    raise ValueError(dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--dataset", choices=["meld", "iemocap", "all"], default="all")
    parser.add_argument("--keep-unsupported", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    datasets = ["meld", "iemocap"] if args.dataset == "all" else [args.dataset]
    summaries = {}

    for dataset in datasets:
        input_dir, output_dir = dataset_paths(project_root, dataset)
        output_dir.mkdir(parents=True, exist_ok=True)
        split_summaries = {}
        for split in ("train", "dev", "test"):
            split_summaries[split] = convert_split(
                dataset,
                input_dir / f"{split}.jsonl",
                output_dir / f"{split}.jsonl",
                drop_unsupported=not args.keep_unsupported,
            )
        labels = LABEL_ORDERS[dataset]
        (output_dir / "labels.json").write_text(
            json.dumps({"emotion": labels}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summary = {
            "dataset": dataset,
            "prompt_style_source": "datasets_video_text/data/new_prompt",
            "gold_field": "emotion_prompt",
            "prompt_field": "qwen_prompt_new",
            "labels": labels,
            "splits": split_summaries,
        }
        (output_dir / "new_prompt_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summaries[dataset] = summary

    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
