#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path


DEFAULT_PROJECT_ROOT = Path(
    "/XYFS01/HDD_POOL/hitsz_mszhang/hitsz_mszhang_1/MRC/MRC/MRC_project/others/AAA/vlm/Graduation-design"
)


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


def summarize(path, check_exists=False):
    rows = list(read_jsonl(path))
    labels = Counter(row.get("emotion") for row in rows)
    missing_video = []
    missing_text = []
    missing_prompt = []
    not_exists = []
    path_patterns = Counter()

    for row in rows:
        sample_id = row.get("sample_id")
        video_path = row.get("video_path")
        if not video_path:
            missing_video.append(sample_id)
        elif check_exists and not Path(video_path).exists():
            not_exists.append({"sample_id": sample_id, "video_path": video_path})
        if not row.get("text"):
            missing_text.append(sample_id)
        if not row.get("qwen_prompt"):
            missing_prompt.append(sample_id)
        if isinstance(video_path, str):
            if "/train_splits/" in video_path:
                path_patterns["meld_train_splits"] += 1
            if "/dev_splits" in video_path:
                path_patterns["meld_dev_splits"] += 1
            if "/output_repeated_splits_test/" in video_path or "/test_splits" in video_path:
                path_patterns["meld_test_splits"] += 1
            if "/sentence_videos/" in video_path:
                path_patterns["iemocap_sentence_videos"] += 1
            if video_path.endswith(".avi"):
                path_patterns["avi"] += 1
            if video_path.endswith(".mp4"):
                path_patterns["mp4"] += 1

    return {
        "path": str(path),
        "rows": len(rows),
        "emotion_counts": dict(sorted(labels.items())),
        "missing_video_count": len(missing_video),
        "missing_video_examples": missing_video[:10],
        "missing_text_count": len(missing_text),
        "missing_prompt_count": len(missing_prompt),
        "not_exists_count": len(not_exists),
        "not_exists_examples": not_exists[:10],
        "path_patterns": dict(sorted(path_patterns.items())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--check-exists", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    manifests = [
        project_root / "datasets_video_text/data/meld/processed/test.jsonl",
        project_root / "datasets_video_text/data/meld/processed/dev.jsonl",
        project_root / "datasets_video_text/data/iemocap/processed_sentence/test.jsonl",
        project_root / "datasets_video_text/data/iemocap/processed_sentence/dev.jsonl",
    ]
    result = [summarize(path, check_exists=args.check_exists) for path in manifests if path.exists()]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

