#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


EVAL_RE = re.compile(
    r"^\[(?P<start>[0-9.]+)\s*-\s*(?P<end>[0-9.]+)\]\s+"
    r"(?P<utt>\S+)\s+(?P<emotion>\S+)\s+\[(?P<vad>[^\]]+)\]"
)

TRANS_RE = re.compile(r"^(?P<utt>\S+)\s+\[(?P<times>[^\]]+)\]:\s*(?P<text>.*)$")

SESSION_TO_SPLIT = {
    "Session1": "train",
    "Session2": "train",
    "Session3": "train",
    "Session4": "dev",
    "Session5": "test",
}

EMOTION_MAP = {
    "ang": "anger",
    "hap": "happiness",
    "exc": "excitement",
    "sad": "sadness",
    "fru": "frustration",
    "fea": "fear",
    "sur": "surprise",
    "neu": "neutral",
    "dis": "disgust",
    "oth": "other",
    "xxx": "unknown",
}


def iter_files(root: Path, *parts):
    for path in root.rglob("*"):
        lowered = str(path).lower().replace("\\", "/")
        if path.is_file() and all(part.lower() in lowered for part in parts):
            yield path


def parse_transcriptions(root: Path):
    transcripts = {}
    for path in iter_files(root, "transcriptions"):
        if path.suffix.lower() != ".txt":
            continue
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                match = TRANS_RE.match(line.strip())
                if match:
                    transcripts[match.group("utt")] = match.group("text").strip()
    return transcripts


def parse_evaluations(root: Path):
    records = {}
    for path in iter_files(root, "emoevaluation"):
        if path.suffix.lower() != ".txt":
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            match = EVAL_RE.match(line.strip())
            if not match:
                continue
            utt = match.group("utt")
            code = match.group("emotion")
            session = next((part for part in path.parts if part.startswith("Session")), None)
            dialogue_id = "_".join(utt.split("_")[:-1])
            utterance_id = utt.split("_")[-1]
            records[utt] = {
                "session": session,
                "dialogue_id": dialogue_id,
                "utterance_id": utterance_id,
                "emotion_code": code,
                "emotion": EMOTION_MAP.get(code, code),
                "start": float(match.group("start")),
                "end": float(match.group("end")),
                "vad": [float(x.strip()) for x in match.group("vad").split(",")],
            }
    return records


def build_video_index(root: Path):
    index = defaultdict(list)
    for ext in ("*.avi", "*.mp4", "*.mov", "*.mkv"):
        for path in root.rglob(ext):
            name = path.stem
            index[name].append(path.resolve())
    return index


def resolve_video(video_index, utterance_id, dialogue_id):
    candidates = [
        utterance_id,
        dialogue_id,
    ]
    for candidate in candidates:
        paths = video_index.get(candidate)
        if paths:
            sentence_level = [p for p in paths if utterance_id in str(p)]
            chosen = sentence_level[0] if sentence_level else paths[0]
            return str(chosen)
    for key, paths in video_index.items():
        if utterance_id in key:
            return str(paths[0])
    return None


def speaker_from_utterance(utt):
    parts = utt.split("_")
    return parts[-1][0] if parts and parts[-1] else None


def make_prompt(context, speaker, text, labels):
    context_lines = [f"{turn['speaker']}: {turn['text']}" for turn in context[-8:]]
    context_text = "\n".join(context_lines) if context_lines else "(none)"
    return (
        "You are given a short IEMOCAP video clip and its dialogue transcript. "
        "Predict the emotion of the target utterance.\n"
        f"Candidate emotions: {', '.join(labels)}\n"
        f"Previous context:\n{context_text}\n"
        f"Target speaker: {speaker}\n"
        f"Target utterance: {text}\n"
        "Answer with exactly one candidate emotion."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--context-window", type=int, default=8)
    parser.add_argument(
        "--keep-unknown",
        action="store_true",
        help="Keep IEMOCAP xxx/unknown labels instead of dropping them.",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    transcripts = parse_transcriptions(raw_dir)
    evaluations = parse_evaluations(raw_dir)
    video_index = build_video_index(raw_dir)

    records = []
    for utt, item in evaluations.items():
        if item["emotion"] == "unknown" and not args.keep_unknown:
            continue
        text = transcripts.get(utt)
        if not text:
            continue
        session = item["session"]
        split = SESSION_TO_SPLIT.get(session, "train")
        dialogue_id = item["dialogue_id"]
        video_path = resolve_video(video_index, utt, dialogue_id)
        records.append(
            {
                "dataset": "iemocap",
                "split": split,
                "sample_id": f"iemocap_{utt}",
                "session": session,
                "dialogue_id": dialogue_id,
                "utterance_id": item["utterance_id"],
                "speaker": speaker_from_utterance(utt),
                "text": text,
                "emotion": item["emotion"],
                "emotion_code": item["emotion_code"],
                "sentiment": None,
                "video_path": video_path,
                "start": item["start"],
                "end": item["end"],
                "vad": item["vad"],
            }
        )

    labels = sorted({record["emotion"] for record in records})
    by_split_dialogue = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_split_dialogue[record["split"]][record["dialogue_id"]].append(record)

    summary = {"dataset": "iemocap", "splits": {}, "missing_videos": {}}
    for split in ("train", "dev", "test"):
        out_path = output_dir / f"{split}.jsonl"
        total = 0
        missing_videos = 0
        emotion_counter = Counter()
        with out_path.open("w", encoding="utf-8") as out:
            for dialogue_id in sorted(by_split_dialogue[split]):
                rows = sorted(by_split_dialogue[split][dialogue_id], key=lambda r: r["start"])
                history = []
                for row in rows:
                    context = history[-args.context_window :]
                    if row["video_path"] is None:
                        missing_videos += 1
                    row = dict(row)
                    row["context"] = context
                    row["qwen_prompt"] = make_prompt(context, row["speaker"], row["text"], labels)
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    history.append(
                        {
                            "speaker": row["speaker"],
                            "text": row["text"],
                            "emotion": row["emotion"],
                        }
                    )
                    emotion_counter[row["emotion"]] += 1
                    total += 1
        summary["splits"][split] = {
            "samples": total,
            "emotion_counts": dict(sorted(emotion_counter.items())),
        }
        summary["missing_videos"][split] = missing_videos

    (output_dir / "labels.json").write_text(
        json.dumps({"emotion": labels}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

