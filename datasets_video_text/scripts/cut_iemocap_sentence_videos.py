#!/usr/bin/env python3
import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SPLITS = ("train", "dev", "test")


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


def clip_output_path(video_root, row):
    split = row["split"]
    session = row.get("session") or "unknown_session"
    dialogue_id = row["dialogue_id"]
    sample_id = row["sample_id"].replace("/", "_")
    return video_root / split / session / dialogue_id / f"{sample_id}.mp4"


def build_ffmpeg_cmd(ffmpeg, row, output_path, keep_audio, overwrite):
    source = row.get("source_video_path") or row.get("video_path")
    if not source:
        raise ValueError(f"{row.get('sample_id')} has empty video_path")
    start = float(row["start"])
    end = float(row["end"])
    duration = max(0.01, end - start)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-ss",
        f"{start:.4f}",
        "-i",
        source,
        "-t",
        f"{duration:.4f}",
        "-map",
        "0:v:0",
    ]
    if keep_audio:
        cmd += ["-map", "0:a:0?", "-c:a", "aac", "-b:a", "96k"]
    else:
        cmd += ["-an"]
    cmd += [
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    return cmd


def cut_one(ffmpeg, row, output_path, keep_audio, overwrite, dry_run):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0 and not overwrite:
        return {"status": "exists", "sample_id": row["sample_id"], "output": str(output_path)}
    cmd = build_ffmpeg_cmd(ffmpeg, row, output_path, keep_audio, overwrite)
    if dry_run:
        return {"status": "dry_run", "sample_id": row["sample_id"], "output": str(output_path)}
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        return {
            "status": "failed",
            "sample_id": row["sample_id"],
            "output": str(output_path),
            "error": result.stderr.strip(),
        }
    return {"status": "created", "sample_id": row["sample_id"], "output": str(output_path)}


def main():
    parser = argparse.ArgumentParser(
        description="Cut IEMOCAP dialogue-level videos into utterance/sentence-level clips."
    )
    parser.add_argument(
        "--input-dir",
        default="/XYFS01/HDD_POOL/hitsz_mszhang/hitsz_mszhang_1/MRC/MRC/MRC_project/others/AAA/vlm/Graduation-design/datasets_video_text/data/iemocap/processed",
        help="Directory containing train/dev/test JSONL manifests from prepare_iemocap.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="/XYFS01/HDD_POOL/hitsz_mszhang/hitsz_mszhang_1/MRC/MRC/MRC_project/others/AAA/vlm/Graduation-design/datasets_video_text/data/iemocap/processed_sentence",
        help="Directory for updated JSONL manifests with sentence-level video_path.",
    )
    parser.add_argument(
        "--video-root",
        default="/XYFS01/HDD_POOL/hitsz_mszhang/hitsz_mszhang_1/MRC/MRC/MRC_project/others/AAA/vlm/Graduation-design/datasets_video_text/data/iemocap/sentence_videos",
        help="Directory for generated sentence-level mp4 clips.",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-audio", action="store_true", help="Keep audio in clips. Default is video-only.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    video_root = Path(args.video_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_root.mkdir(parents=True, exist_ok=True)

    all_tasks = []
    updated_by_split = {}
    for split in SPLITS:
        input_path = input_dir / f"{split}.jsonl"
        rows = list(read_jsonl(input_path))
        updated_rows = []
        for row in rows:
            output_path = clip_output_path(video_root, row)
            updated = dict(row)
            updated["source_video_path"] = row.get("source_video_path") or row.get("video_path")
            updated["video_path"] = str(output_path)
            updated["video_is_sentence_clip"] = True
            updated_rows.append(updated)
            all_tasks.append((row, output_path))
        updated_by_split[split] = updated_rows

    status_counts = {}
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                cut_one,
                args.ffmpeg,
                row,
                output_path,
                args.keep_audio,
                args.overwrite,
                args.dry_run,
            )
            for row, output_path in all_tasks
        ]
        for idx, future in enumerate(as_completed(futures), 1):
            result = future.result()
            status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1
            if result["status"] == "failed":
                failures.append(result)
            if idx % 200 == 0:
                print(f"[IEMOCAP] Processed {idx}/{len(futures)} clips: {status_counts}")

    if failures:
        failure_path = output_dir / "cut_failures.json"
        failure_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(f"{len(failures)} ffmpeg jobs failed. See {failure_path}")

    for split, rows in updated_by_split.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)

    for extra_name in ("labels.json", "dataset_summary.json"):
        src = input_dir / extra_name
        if src.exists():
            dst = output_dir / extra_name
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "video_root": str(video_root),
        "keep_audio": args.keep_audio,
        "dry_run": args.dry_run,
        "status_counts": status_counts,
        "manifest_rows": {split: len(rows) for split, rows in updated_by_split.items()},
    }
    (output_dir / "sentence_video_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

