#!/usr/bin/env python3
import argparse
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
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
    return [normalize_label(label) for label in (data.get("emotion") or data.get("labels"))]


def normalize_label(label):
    return re.sub(r"\s+", " ", str(label or "").strip().lower())


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def clean_text(text):
    return str(text or "").replace("\r", " ").replace("\n", " ").strip()


def speaker_name(row):
    return clean_text(row.get("speaker") or row.get("speaker_original") or "Speaker")


def map_context_speaker(row, speaker):
    speaker_map = row.get("speaker_map") or {}
    return speaker_map.get(str(speaker), str(speaker))


def conversation_text(row):
    context = row.get("context") or []
    lines = []
    for turn in context:
        speaker = map_context_speaker(row, turn.get("speaker", "Speaker"))
        text = clean_text(turn.get("text", ""))
        if text:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines) if lines else "(none)"


def extract_label(output_text, labels):
    labels = [normalize_label(label) for label in labels]
    text = normalize_text(output_text)
    match = re.search(r"final_answer\s*:\s*([a-zA-Z_ -]+)", text)
    candidates = [match.group(1).strip() if match else "", text.strip(" .,:;\"'`[](){}")]
    for candidate in candidates:
        candidate = normalize_label(candidate)
        if candidate in labels:
            return candidate
    for label in labels:
        for alias in ALIASES.get(label, [label]):
            pattern = r"(?<![a-z])" + re.escape(alias.lower()) + r"(?![a-z])"
            if re.search(pattern, text):
                return label
    return None


def extract_reason(output_text, field_name):
    text = (output_text or "").strip()
    if not text:
        return "", "empty"
    text = re.sub(r"^```(?:text)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    pattern = rf"{re.escape(field_name)}\s*:\s*(.*)"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return text, "ok_unparsed"
    reason = match.group(1).strip()
    for marker in (
        "OBSERVATION",
        "CONTEXT_REASON",
        "VISUAL_REASON",
        "DIALOGUE_REASON",
        "FINAL_REASON",
        "FINAL_ANSWER",
    ):
        if marker.lower() == field_name.lower():
            continue
        marker_match = re.search(rf"\n\s*{marker}\s*:", reason, flags=re.IGNORECASE)
        if marker_match:
            reason = reason[: marker_match.start()].strip()
            break
    return reason, "ok" if reason else "empty"


def visual_prompt(row):
    return (
        "You are constructing visually grounded evidence for multimodal emotion recognition in conversation.\n\n"
        "You are given sampled video frames from the current utterance video clip. "
        "The text below is provided only to help identify the target speaker and locate "
        "the current speaking moment. Do not use the meaning of the utterance as "
        "emotional evidence in this step.\n\n"
        "### Target Speaker\n"
        f"{speaker_name(row)}\n\n"
        "### Current Utterance\n"
        f"\"{clean_text(row.get('text', ''))}\"\n\n"
        "### Task\n"
        "Write a compact but detailed visual evidence paragraph that can later be used "
        "as part of a reasoning chain for emotion recognition.\n\n"
        "Your description should focus on observable visual cues of the target speaker. "
        "When visible, analyze:\n\n"
        "* whether the target speaker can be identified in the frames;\n"
        "* facial details: eyes, eyebrows, mouth, smile, frown, jaw tension, facial relaxation or stiffness;\n"
        "* gaze and head behavior: eye contact, looking away, looking down, head tilt, nodding, shaking, turning;\n"
        "* body behavior: posture, hand gestures, shoulder tension, leaning forward/backward, stillness, movement, interpersonal distance;\n"
        "* temporal changes across the sampled frames, such as expression appearing, fading, intensifying, or shifting;\n"
        "* interaction and scene context, including other people's reactions only when they help interpret the target speaker's visible behavior.\n\n"
        "Rules:\n"
        "* Do not predict the final emotion label.\n"
        "* Do not output candidate labels.\n"
        "* Do not use the dialogue text as emotional evidence.\n"
        "* Do not invent facial expressions, gestures, movements, or speaker identity when they are not visible.\n"
        "* Prefer concrete visual observations over abstract emotion words.\n"
        "* You may cautiously describe affective implications such as \"tense\", \"relaxed\", \"hesitant\", \"guarded\", \"animated\", \"withdrawn\", or \"high-arousal\", but avoid directly naming a final emotion category.\n"
        "* If the target speaker is not clearly visible, not identifiable, occluded, too small, or visually ambiguous, state this explicitly and describe only reliable visible cues.\n"
        "* If the visual evidence is weak, subtle, or neutral-looking, say so explicitly instead of forcing an emotional interpretation.\n"
        "* Write one coherent paragraph of about 80 to 180 words.\n\n"
        "Output exactly this schema:\n\n"
        "VISUAL_REASON: ..."
    )


def dialogue_prompt(row, labels, gold):
    label_text = ", ".join(labels)
    return (
        "You are constructing dialogue-only reasoning data for emotion recognition in conversation.\n\n"
        "You will be given a dialogue context, the current speaker, the current utterance, "
        "and the target emotion label. Your task is to explain why the target emotion is "
        "reasonable based only on the dialogue text.\n\n"
        "### Dialogue Context\n"
        f"{conversation_text(row)}\n\n"
        "### Current Speaker\n"
        f"{speaker_name(row)}\n\n"
        "### Current Utterance\n"
        f"\"{clean_text(row.get('text', ''))}\"\n\n"
        "### Target Emotion Label\n"
        f"{gold}\n\n"
        "### Candidate Emotion Labels\n"
        f"{label_text}\n\n"
        "### Task\n"
        f"Write a dialogue-only reasoning explanation for why the current speaker's emotion is \"{gold}\".\n\n"
        "Analysis requirements:\n"
        "- Use only the dialogue text and conversational context.\n"
        "- Do not mention visual, audio, facial expression, body gesture, gaze, or scene evidence.\n"
        "- Explain lexical evidence, pragmatic meaning, speaker intention, discourse structure, and emotional shift when relevant.\n"
        "- Explicitly refer to or quote key words from the current utterance or previous turns.\n"
        "- If the emotion is subtle, explain why it is implied rather than directly stated.\n"
        "- Do not say that a gold label is provided.\n"
        "- Write one coherent paragraph, about 80 to 200 words.\n\n"
        "Output exactly this schema:\n\n"
        "DIALOGUE_REASON: ..."
    )


def predict_prompt(row, labels, prompt_field):
    base_prompt = row.get(prompt_field) or (
        f"Dialogue context:\n{conversation_text(row)}\n\n"
        f"Current speaker: {speaker_name(row)}\n"
        f"Current utterance: \"{clean_text(row.get('text', ''))}\""
    )
    label_text = ", ".join(labels)
    return (
        f"{base_prompt}\n\n"
        "Diagnostic prediction mode: use the available video and dialogue evidence to predict the emotion.\n"
        "Output exactly this schema:\n"
        "FINAL_REASON: one concise sentence.\n"
        f"FINAL_ANSWER: exactly one lowercase label from this list: {label_text}."
    )


def build_prompt(row, labels, prompt_field, step, gold=None):
    if step == "visual":
        return visual_prompt(row)
    if step == "dialogue":
        return dialogue_prompt(row, labels, gold or "")
    if step == "predict":
        return predict_prompt(row, labels, prompt_field)
    raise ValueError(f"Unknown step: {step}")


def file_url(path):
    return "file://" + str(Path(path).resolve())


def build_messages(row, labels, args):
    prompt = build_prompt(row, labels, args.prompt_field, args.step, args.current_gold)
    content = []
    if args.step in ("visual", "predict"):
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


def existing_sample_ids(path, step):
    if not Path(path).exists():
        return set()
    ids = set()
    for row in read_jsonl(path):
        sample_id = row.get("sample_id")
        if not sample_id:
            continue
        if step == "visual" and row.get("visual_reason"):
            ids.add(sample_id)
        elif step == "dialogue" and row.get("dialogue_reason"):
            ids.add(sample_id)
        elif step == "predict" and row.get("prediction"):
            ids.add(sample_id)
    return ids


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
    if step == "visual":
        key = "step1_visual_reason"
    elif step == "dialogue":
        key = "step2_dialogue_reason"
    else:
        key = "diagnostic_predict"
    root = resolve_project_path(cfg, cfg["output"]["root"]) / cfg["output"].get(key, key)
    return root / dataset / f"{split}_shard{shard}.jsonl"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--step", choices=["visual", "dialogue", "predict"], required=True)
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


def base_record(row, args, gold, status, raw_output, raw_response, server, started, error):
    return {
        "sample_id": row.get("sample_id"),
        "dataset": args.dataset,
        "split": args.split,
        "step": args.step,
        "status": status,
        "gold": gold,
        "teacher_output": raw_output,
        "server": server,
        "latency_sec": round(time.time() - started, 4),
        "error": error,
        "source": {
            "video_path": row.get("video_path"),
            "dialogue_id": row.get("dialogue_id"),
            "utterance_id": row.get("utterance_id"),
            "speaker": row.get("speaker"),
            "speaker_original": row.get("speaker_original"),
            "text": row.get("text"),
            "conversation": conversation_text(row),
        },
        "raw_response_id": raw_response.get("id") if isinstance(raw_response, dict) else None,
    }


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
    server_for_shard = servers[args.shard_index % len(servers)]
    done_ids = existing_sample_ids(args.output, args.step) if args.resume else set()
    counters = Counter(total_samples=len(rows), resumed=len(done_ids))

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
                "server_for_shard": server_for_shard,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for local_idx, row in enumerate(rows, 1):
        sample_id = row.get("sample_id", f"{args.dataset}_{args.split}_{local_idx}")
        if sample_id in done_ids:
            continue
        started = time.time()
        gold = normalize_label(row.get(args.gold_field, ""))
        args.current_gold = gold
        if args.step in ("visual", "predict") and (not row.get("video_path") or not Path(row["video_path"]).exists()):
            counters["missing_video"] += 1
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
            raw_output, raw_response = generate_one(row, labels, args, server_for_shard)
            error = None
            if args.step == "visual":
                reason, status = extract_reason(raw_output, "VISUAL_REASON")
                record = base_record(row, args, gold, status, raw_output, raw_response, server_for_shard, started, error)
                record["visual_reason"] = reason
                counters["visual_reason_generated" if reason else "visual_reason_empty"] += 1
            elif args.step == "dialogue":
                reason, status = extract_reason(raw_output, "DIALOGUE_REASON")
                record = base_record(row, args, gold, status, raw_output, raw_response, server_for_shard, started, error)
                record["dialogue_reason"] = reason
                counters["dialogue_reason_generated" if reason else "dialogue_reason_empty"] += 1
            else:
                prediction = extract_label(raw_output, labels)
                status = "ok" if prediction else "ok_unparsed"
                record = base_record(row, args, gold, status, raw_output, raw_response, server_for_shard, started, error)
                record["prediction"] = prediction
                record["correct"] = prediction == gold
                counters["prediction_generated" if prediction else "prediction_unparsed"] += 1
        except Exception as exc:
            counters["failed"] += 1
            record = base_record(row, args, gold, "failed", "", None, server_for_shard, started, repr(exc))

        counters[record["status"]] += 1
        append_jsonl(args.output, record)
        if local_idx % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "processed_shard_rows": local_idx,
                        "total_shard_rows": len(rows),
                        "counters": dict(sorted(counters.items())),
                    }
                ),
                flush=True,
            )

    print(json.dumps({"event": "summary", "counters": dict(sorted(counters.items()))}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
