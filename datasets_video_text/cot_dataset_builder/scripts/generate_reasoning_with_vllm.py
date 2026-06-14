#!/usr/bin/env python3
import argparse
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from config_utils import load_config, resolve_project_path


ALIASES = {
    "anger": ["anger", "angry"],
    "angry": ["angry", "anger"],
    "disgust": ["disgust", "disgusted"],
    "fear": ["fear", "fearful", "scared"],
    "frustration": ["frustration", "frustrated"],
    "frustrated": ["frustrated", "frustration"],
    "happiness": ["happiness", "happy"],
    "happy": ["happy", "happiness"],
    "joy": ["joy", "joyful", "happy"],
    "joyful": ["joyful", "joy", "happy"],
    "neutral": ["neutral"],
    "sadness": ["sadness", "sad"],
    "sad": ["sad", "sadness"],
    "surprise": ["surprise", "surprised"],
    "excitement": ["excitement", "excited"],
    "excited": ["excited", "excitement"],
    "other": ["other"],
}


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def load_labels(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [str(label).lower() for label in (data.get("emotion") or data.get("labels"))]


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def extract_label(output_text, labels):
    labels = [label.lower() for label in labels]
    text = normalize_text(output_text)
    match = re.search(r"final_answer\s*:\s*([a-zA-Z_ -]+)", text)
    candidates = [match.group(1).strip() if match else "", text.strip(" .,:;\"'`[](){}")]
    for candidate in candidates:
        if candidate in labels:
            return candidate
    for label in labels:
        for alias in ALIASES.get(label, [label]):
            pattern = r"(?<![a-z])" + re.escape(alias.lower()) + r"(?![a-z])"
            if re.search(pattern, text):
                return label
    return None


def clean_text(text):
    return str(text or "").replace("\r", " ").replace("\n", " ").strip()


def dialogue_lines(row, prompt_field):
    if row.get(prompt_field):
        return row[prompt_field]
    context = row.get("context") or []
    parts = []
    for turn in context:
        parts.append(f'{turn.get("speaker", "Speaker")}: {clean_text(turn.get("text", ""))}')
    parts.append(f'Target speaker: {row.get("speaker", "")}')
    parts.append(f'Target utterance: {clean_text(row.get("text", ""))}')
    return "\n".join(parts)


def build_prompt(row, labels, prompt_field, step):
    label_text = ", ".join(labels)
    target = f'{row.get("speaker", "")}: "{clean_text(row.get("text", ""))}"'
    base_prompt = dialogue_lines(row, prompt_field)
    if step == "visual":
        evidence_scope = (
            "Use both the sampled video frames and the dialogue text. "
            "If the target speaker is not visually identifiable, say so explicitly."
        )
        visual_rule = (
            "VISUAL_REASON must include speaker_visible=<yes|no|uncertain> and "
            "visual_confidence=<high|medium|low>, then explain the visible facial "
            "expression, body pose, gaze, movement, scene context, or absence of usable cues."
        )
    else:
        evidence_scope = "Use only the dialogue text. Do not invent visual evidence."
        visual_rule = (
            "DIALOGUE_REASON should mention text-only confidence=<high|medium|low> "
            "and explain lexical, pragmatic, and conversational-context evidence."
        )

    return (
        "You are building high-quality reasoning data for multimodal emotion recognition.\n"
        f"{evidence_scope}\n"
        "The reference prompt style below is the zero-shot prompt that performed best in prior experiments.\n\n"
        f"[Reference zero-shot prompt]\n{base_prompt}\n\n"
        f"[Target utterance]\n{target}\n\n"
        f"[Candidate labels]\n{label_text}\n\n"
        "Write a compact but faithful teacher answer for SFT/RL data construction.\n"
        "Do not reveal hidden annotations or dataset metadata. Do not say that a gold label is given.\n"
        f"{visual_rule}\n"
        "Your output must use exactly this plain-text schema:\n"
        "OBSERVATION: one or two sentences describing the directly available evidence.\n"
        "CONTEXT_REASON: one or two sentences explaining how the dialogue context affects the target utterance.\n"
        f"{'VISUAL_REASON' if step == 'visual' else 'DIALOGUE_REASON'}: one or two sentences with the required confidence marker.\n"
        "FINAL_REASON: one sentence connecting the evidence to the selected emotion.\n"
        "FINAL_ANSWER: exactly one lowercase label from the candidate labels."
    )


def file_url(path):
    return "file://" + str(Path(path).resolve())


def build_messages(row, labels, args):
    prompt = build_prompt(row, labels, args.prompt_field, args.step)
    content = []
    if args.step == "visual":
        content.append({"type": "video_url", "video_url": {"url": file_url(row["video_path"])}})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def post_chat_completion(base_url, payload, timeout):
    url = base_url.rstrip("/") + "/chat/completions"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def generate_one(row, labels, args, server):
    payload = {
        "model": args.model,
        "messages": build_messages(row, labels, args),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "presence_penalty": args.presence_penalty,
        "extra_body": {
            "top_k": args.top_k,
            "mm_processor_kwargs": {
                "fps": args.fps,
                "do_sample_frames": args.do_sample_frames,
            },
        },
    }
    last_error = None
    for attempt in range(1, args.max_retries + 1):
        try:
            response = post_chat_completion(server, payload, args.timeout_sec)
            return response["choices"][0]["message"].get("content", ""), response
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            last_error = repr(exc)
            time.sleep(min(30, 2**attempt))
    raise RuntimeError(last_error)


def existing_sample_ids(path):
    if not Path(path).exists():
        return set()
    return {row.get("sample_id") for row in read_jsonl(path) if row.get("sample_id")}


def dataset_cfg(cfg, dataset):
    for item in cfg["data"]["datasets"]:
        if item["name"] == dataset:
            return item
    raise ValueError(f"Unknown dataset in config: {dataset}")


def default_manifest(cfg, dataset, split):
    item = dataset_cfg(cfg, dataset)
    return resolve_project_path(cfg, item["manifest_dir"]) / f"{split}.jsonl"


def default_label_file(cfg, dataset):
    return resolve_project_path(cfg, dataset_cfg(cfg, dataset)["label_file"])


def default_output(cfg, step, dataset, split, shard):
    step_key = "step1_visual_reason" if step == "visual" else "step2_dialogue_reason"
    root = resolve_project_path(cfg, cfg["output"]["root"]) / cfg["output"][step_key]
    return root / dataset / f"{split}_shard{shard}.jsonl"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--step", choices=["visual", "dialogue"], required=True)
    parser.add_argument("--dataset", choices=["meld", "iemocap"], required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--manifest")
    parser.add_argument("--label-file")
    parser.add_argument("--output")
    parser.add_argument("--servers", required=True, help="Comma-separated OpenAI base URLs, e.g. http://127.0.0.1:18000/v1,http://127.0.0.1:18001/v1")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(args.config)
    gen = cfg["generation"]
    data_item = dataset_cfg(cfg, args.dataset)
    args.model = cfg["model"].get("served_model_name", cfg["model"]["name"])
    args.prompt_field = data_item.get("prompt_field", "qwen_prompt_new")
    args.gold_field = data_item.get("gold_field", "emotion_prompt")
    args.manifest = args.manifest or str(default_manifest(cfg, args.dataset, args.split))
    args.label_file = args.label_file or str(default_label_file(cfg, args.dataset))
    args.output = args.output or str(default_output(cfg, args.step, args.dataset, args.split, args.shard_index))
    args.fps = float(gen.get("fps", 2))
    args.do_sample_frames = bool(gen.get("do_sample_frames", True))
    args.max_tokens = int(gen.get("max_tokens", 4096))
    args.temperature = float(gen.get("temperature", 1.0))
    args.top_p = float(gen.get("top_p", 0.95))
    args.top_k = int(gen.get("top_k", 20))
    args.presence_penalty = float(gen.get("presence_penalty", 0.0))
    args.timeout_sec = int(gen.get("timeout_sec", 900))
    args.max_retries = int(gen.get("max_retries", 3))
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    labels = load_labels(args.label_file)
    rows = list(read_jsonl(args.manifest))
    rows = [row for idx, row in enumerate(rows) if idx % args.num_shards == args.shard_index]
    if args.limit > 0:
        rows = rows[: args.limit]

    servers = [server.strip() for server in args.servers.split(",") if server.strip()]
    if not servers:
        raise ValueError("--servers cannot be empty")
    done_ids = existing_sample_ids(args.output) if args.resume else set()

    print(
        json.dumps(
            {
                "event": "start",
                "step": args.step,
                "dataset": args.dataset,
                "split": args.split,
                "rows_shard": len(rows),
                "done": len(done_ids),
                "output": args.output,
                "servers": servers,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for local_idx, row in enumerate(rows, 1):
        sample_id = row.get("sample_id", f"{args.dataset}_{args.split}_{local_idx}")
        if sample_id in done_ids:
            continue
        server = servers[(local_idx - 1) % len(servers)]
        started = time.time()
        gold = str(row.get(args.gold_field, "")).lower()
        if args.step == "visual" and (not row.get("video_path") or not Path(row["video_path"]).exists()):
            append_jsonl(
                args.output,
                {
                    "sample_id": sample_id,
                    "dataset": args.dataset,
                    "split": args.split,
                    "step": args.step,
                    "status": "skipped_missing_video",
                    "gold": gold,
                    "video_path": row.get("video_path"),
                },
            )
            continue
        try:
            raw_output, raw_response = generate_one(row, labels, args, server)
            prediction = extract_label(raw_output, labels)
            status = "ok" if prediction else "ok_unparsed"
            error = None
        except Exception as exc:
            raw_output = ""
            raw_response = None
            prediction = None
            status = "failed"
            error = repr(exc)

        append_jsonl(
            args.output,
            {
                "sample_id": sample_id,
                "dataset": args.dataset,
                "split": args.split,
                "step": args.step,
                "status": status,
                "gold": gold,
                "prediction": prediction,
                "correct": prediction == gold,
                "teacher_output": raw_output,
                "server": server,
                "latency_sec": round(time.time() - started, 4),
                "error": error,
                "source": {
                    "video_path": row.get("video_path"),
                    "dialogue_id": row.get("dialogue_id"),
                    "utterance_id": row.get("utterance_id"),
                    "speaker": row.get("speaker"),
                    "text": row.get("text"),
                    "prompt": row.get(args.prompt_field),
                },
                "raw_response_id": raw_response.get("id") if isinstance(raw_response, dict) else None,
            },
        )
        if local_idx % args.log_every == 0:
            print(json.dumps({"event": "progress", "processed_shard_rows": local_idx, "total_shard_rows": len(rows)}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
