#!/usr/bin/env python3
import argparse
import glob
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

INSTRUCTION_LEAK_PATTERNS = (
    r"\bgold label is provided\b",
    r"\btarget label is provided\b",
    r"\baccording to the prompt\b",
    r"\bthe task asks me\b",
    r"\bthe prompt asks\b",
    r"\bas instructed\b",
)

FUSION_KEYWORDS = {
    "speaker_not_visible": (
        "not clearly visible",
        "cannot be identified",
        "off-camera",
        "occluded",
        "too small",
        "visually ambiguous",
    ),
    "visual_weak_or_limited": (
        "weak",
        "limited support",
        "little support",
        "subtle",
        "neutral-looking",
        "provides limited",
        "little or no speaker-specific evidence",
    ),
    "visual_conflict_or_not_aligned": (
        "inconsistent",
        "does not clearly support",
        "does not support",
        "not aligned",
        "conflict",
        "more decisive",
    ),
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


def speaker_display(row):
    speaker = clean_text(row.get("speaker") or "")
    original = clean_text(row.get("speaker_original") or "")
    if original and speaker and original != speaker:
        return f"{original} ({speaker} in the transcript)"
    return original or speaker or "Speaker"


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


def extract_reason(output_text, field_name, strict_schema=False):
    text = (output_text or "").strip()
    if not text:
        return "", "empty"
    text = re.sub(r"^```(?:text)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    pattern = rf"{re.escape(field_name)}\s*:\s*(.*)"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        if strict_schema:
            return "", "ok_unparsed"
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


def parse_dialogue_reason(output_text):
    reason, status = extract_reason(output_text, "DIALOGUE_REASON", strict_schema=True)
    if status != "ok" or not reason:
        return None, "parse_failed"
    return reason, "ok"


def has_instruction_leak(text):
    lowered = normalize_text(text)
    return any(re.search(pattern, lowered) for pattern in INSTRUCTION_LEAK_PATTERNS)


def parse_fusion_response(output_text, gold, labels):
    text = (output_text or "").strip()
    if not text:
        return None, None, "parse_failed", "empty_output"
    think_match = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.IGNORECASE | re.DOTALL)
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if not think_match or not answer_match:
        return None, None, "parse_failed", "missing_think_or_answer"
    think = think_match.group(1).strip()
    raw_answer = answer_match.group(1).strip()
    answer = normalize_label(raw_answer)
    if not think:
        return None, answer, "parse_failed", "empty_think"
    if has_instruction_leak(think):
        return think, answer, "parse_failed", "instruction_leak"
    if raw_answer != raw_answer.lower():
        return think, answer, "parse_failed", "answer_not_lowercase"
    if answer not in labels:
        return think, answer, "parse_failed", "answer_not_in_labels"
    if answer != gold:
        return think, answer, "parse_failed", "answer_not_gold"
    return think, answer, "ok", None


def count_fusion_flags(text):
    lowered = normalize_text(text)
    flags = {}
    for name, keywords in FUSION_KEYWORDS.items():
        flags[name] = any(keyword in lowered for keyword in keywords)
    return flags


def visual_prompt(row):
    return (
        "You are generating the final visual evidence text for a multimodal emotion recognition dataset.\n\n"
        "You are given sampled video frames from the current utterance video clip. "
        "The text below is provided only to help identify the target speaker and the "
        "speaking moment. Do not use the meaning of the utterance as emotional evidence.\n\n"
        "### Target Speaker\n"
        f"{speaker_display(row)}\n\n"
        "### Current Utterance\n"
        f"\"{clean_text(row.get('text', ''))}\"\n\n"
        "### Task\n"
        "Describe the observable visual cues of the target speaker that may be useful "
        "for later emotion reasoning.\n\n"
        "Focus only on visible evidence:\n"
        "- target speaker identity and visibility;\n"
        "- facial expression: eyes, eyebrows, mouth, smile, frown, jaw tension, facial stiffness or relaxation;\n"
        "- gaze and head movement;\n"
        "- body posture, hand gesture, stillness, movement, interpersonal distance;\n"
        "- visible changes across the frames;\n"
        "- interaction with other people only when it is visually observable.\n\n"
        "Strict output rules:\n"
        "- Output only one line beginning with \"VISUAL_REASON:\".\n"
        "- Do not write \"Thinking Process\", \"Analysis\", \"Step\", \"Identify\", \"Drafting\", or bullet points.\n"
        "- Do not explain how you solved the task.\n"
        "- Do not predict the final emotion label.\n"
        "- Do not output candidate labels.\n"
        "- Do not use the dialogue meaning as emotional evidence.\n"
        "- Do not quote or interpret the utterance.\n"
        "- Do not invent facial expressions, gestures, movements, or speaker identity.\n"
        "- If the target speaker is not clearly visible or cannot be identified, say so explicitly.\n"
        "- Use concrete visual observations first. You may use cautious affective words such as tense, relaxed, hesitant, guarded, animated, withdrawn, or low-arousal, but avoid naming a final emotion category.\n"
        "- Keep the description between 60 and 130 words.\n\n"
        "Output format:\n"
        "VISUAL_REASON: <one compact paragraph>"
    )


def dialogue_prompt(row, labels, gold):
    label_text = ", ".join(labels)
    return (
        "You are constructing dialogue-only reasoning data for emotion recognition in conversation.\n\n"
        "You will be given a dialogue context, the current speaker, the current utterance, "
        "and the target emotion label. Your task is to write a faithful text-based "
        "explanation for why the target emotion can be assigned to the current speaker "
        "in this utterance.\n\n"
        "Use only the dialogue text and conversational context. Do not use or mention "
        "any visual, audio, facial, body, gaze, posture, or scene evidence.\n\n"
        "### Dialogue Context\n"
        f"{conversation_text(row)}\n\n"
        "### Current Speaker\n"
        f"{speaker_display(row)}\n\n"
        "### Current Utterance\n"
        f"\"{clean_text(row.get('text', ''))}\"\n\n"
        "### Target Emotion Label\n"
        f"{gold}\n\n"
        "### Candidate Emotion Labels\n"
        f"{label_text}\n\n"
        "### Task\n"
        f"Write a dialogue-only reasoning paragraph explaining why the current speaker's emotion can be interpreted as \"{gold}\" from the text and context.\n\n"
        "Requirements:\n"
        "- Use only textual and conversational evidence.\n"
        "- Do not mention video, audio, facial expression, body gesture, gaze, posture, scene, or visual cues.\n"
        "- Refer to or quote key words from the current utterance or previous turns when useful.\n"
        "- Explain the role of the current utterance, previous turns, speaker intention, pragmatic meaning, discourse progression, and emotional shift when relevant.\n"
        "- Pay attention to short utterances, hesitation, repetition, punctuation, rhetorical questions, contrast, sarcasm, jokes, and implicit emotional shifts.\n"
        "- If the textual evidence is indirect, subtle, or weak, state this honestly instead of overstating the evidence.\n"
        "- Do not say that a gold label is provided.\n"
        "- Do not compare all candidate labels mechanically.\n"
        "- Write one coherent paragraph, about 80 to 200 words.\n\n"
        "Output exactly this schema:\n\n"
        "DIALOGUE_REASON: ..."
    )


def fusion_prompt(row, labels, gold, visual_reason, dialogue_reason):
    label_text = ", ".join(labels)
    return (
        "You are constructing final multimodal reasoning data for emotion recognition in conversation.\n\n"
        "You will be given:\n"
        "1. the dialogue context and current utterance;\n"
        "2. a visual evidence paragraph generated from the video;\n"
        "3. a dialogue-only reasoning paragraph generated from the text;\n"
        "4. the target emotion label.\n\n"
        "Your task is to write the final multimodal reasoning response for SFT/RL data construction.\n\n"
        "### Dialogue Context\n"
        f"{conversation_text(row)}\n\n"
        "### Current Speaker\n"
        f"{speaker_display(row)}\n\n"
        "### Current Utterance\n"
        f"\"{clean_text(row.get('text', ''))}\"\n\n"
        "### Candidate Emotion Labels\n"
        f"{label_text}\n\n"
        "### Target Emotion Label\n"
        f"{gold}\n\n"
        "### Visual Evidence\n"
        f"{visual_reason}\n\n"
        "### Dialogue-only Reasoning\n"
        f"{dialogue_reason}\n\n"
        "### Task\n"
        f"Write a faithful multimodal reasoning chain that explains why the final emotion label is \"{gold}\", using the dialogue-only reasoning and the visual evidence appropriately.\n\n"
        "Important reasoning rules:\n"
        f"- The final answer must be exactly \"{gold}\".\n"
        "- Do not say that a gold label or target label is provided.\n"
        "- Do not invent any new visual details beyond the given Visual Evidence.\n"
        "- Do not invent dialogue content beyond the given Dialogue Context and Current Utterance.\n"
        "- Do not force the visual evidence to support the final label if it does not.\n"
        "- If the target speaker is not visible, off-camera, unclear, or visually ambiguous, explicitly state that the video provides little or no speaker-specific evidence and rely mainly on the dialogue context.\n"
        "- If the visual evidence is weak, neutral-looking, or not clearly emotion-related, state that it provides limited support.\n"
        "- If the visual evidence appears inconsistent with the final label, state the inconsistency honestly and explain why the dialogue context is more decisive.\n"
        "- If the visual evidence supports the dialogue reasoning, explain how it reinforces the final interpretation.\n"
        "- The reasoning should mention both modalities, but it should weight them differently depending on their reliability.\n"
        "- Avoid mechanical phrasing. Write a natural, coherent reasoning paragraph.\n"
        "- Keep the reasoning between 120 and 260 words.\n\n"
        "Output exactly in this format:\n\n"
        "<think>\n"
        "...\n"
        "</think>\n"
        "<answer>\n"
        f"{gold}\n"
        "</answer>"
    )


def build_fusion_prompt(row, labels, gold, visual_reason, dialogue_reason):
    return fusion_prompt(row, labels, gold, visual_reason, dialogue_reason)


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
    if args.step == "fusion":
        prompt = build_fusion_prompt(
            row,
            labels,
            args.current_gold,
            args.current_visual_reason,
            args.current_dialogue_reason,
        )
    else:
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


def flatten_content(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(value)


def response_text_and_meta(response):
    choices = response.get("choices") or []
    if not choices:
        return "", {"response_empty_choices": True}
    choice = choices[0]
    message = choice.get("message") or {}
    content = flatten_content(message.get("content")).strip()
    reasoning_content = flatten_content(message.get("reasoning_content"))
    return content, {
        "response_text_source": "content" if content else None,
        "finish_reason": choice.get("finish_reason"),
        "message_keys": sorted(message.keys()),
        "has_reasoning_content": bool(reasoning_content.strip()),
        "reasoning_content_chars": len(reasoning_content),
    }


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
    max_attempts = args.max_retries + args.empty_output_retries
    for attempt in range(1, max_attempts + 1):
        try:
            response = post_chat_completion(server, payload, args.timeout_sec)
            text, response_meta = response_text_and_meta(response)
            if text or attempt > args.empty_output_retries:
                return text, response, response_meta
            last_error = "empty_response_text"
            time.sleep(min(10, 2**attempt))
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
        if step == "visual" and row.get("status") == "ok" and row.get("visual_reason"):
            ids.add(sample_id)
        elif step == "dialogue" and row.get("status") == "ok" and row.get("dialogue_reason"):
            ids.add(sample_id)
        elif step == "fusion" and row.get("status") == "ok" and row.get("fusion_reason") and row.get("final_answer"):
            ids.add(sample_id)
        elif step == "predict" and row.get("status") == "ok" and row.get("prediction"):
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


def default_reason_patterns(cfg, dataset, split, reason_step):
    key = "step1_visual_reason" if reason_step == "visual" else "step2_dialogue_reason"
    root = resolve_project_path(cfg, cfg["output"]["root"]) / cfg["output"][key]
    return [str(root / dataset / f"{split}_shard*.jsonl")]


def load_reason_map(patterns, reason_step):
    reason_key = "visual_reason" if reason_step == "visual" else "dialogue_reason"
    by_id = {}
    for pattern in patterns:
        for file in sorted(glob.glob(str(pattern))):
            for row in read_jsonl(file):
                if row.get("status") != "ok":
                    continue
                sample_id = row.get("sample_id")
                reason = row.get(reason_key)
                if sample_id and reason:
                    by_id[sample_id] = row
    return by_id


def default_output(cfg, step, dataset, split, shard):
    if step == "visual":
        key = "step1_visual_reason"
    elif step == "dialogue":
        key = "step2_dialogue_reason"
    elif step == "fusion":
        key = "step3_fusion_reason"
    else:
        key = "diagnostic_predict"
    root = resolve_project_path(cfg, cfg["output"]["root"]) / cfg["output"].get(key, key)
    return root / dataset / f"{split}_shard{shard}.jsonl"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--step", choices=["visual", "dialogue", "fusion", "predict"], required=True)
    parser.add_argument("--dataset", choices=["meld", "iemocap"], required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--manifest")
    parser.add_argument("--label-file")
    parser.add_argument("--output")
    parser.add_argument("--servers", required=True, help="Comma-separated OpenAI base URLs, e.g. http://127.0.0.1:18000/v1,http://127.0.0.1:18001/v1")
    parser.add_argument("--visual-reason-file", "--visual_reason_file", action="append")
    parser.add_argument("--dialogue-reason-file", "--dialogue_reason_file", action="append")
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
    args.visual_reason_file = args.visual_reason_file or default_reason_patterns(cfg, args.dataset, args.split, "visual")
    args.dialogue_reason_file = args.dialogue_reason_file or default_reason_patterns(cfg, args.dataset, args.split, "dialogue")
    args.fps = float(gen.get("fps", 2))
    args.do_sample_frames = bool(gen.get("do_sample_frames", True))
    step_max_tokens_key = f"max_tokens_{args.step}"
    args.max_tokens = int(gen.get(step_max_tokens_key, gen.get("max_tokens", 1024)))
    args.temperature = float(gen.get("temperature", 1.0))
    args.top_p = float(gen.get("top_p", 0.95))
    args.top_k = int(gen.get("top_k", 20))
    args.presence_penalty = float(gen.get("presence_penalty", 0.0))
    args.timeout_sec = int(gen.get("timeout_sec", 900))
    args.max_retries = int(gen.get("max_retries", 3))
    args.empty_output_retries = int(gen.get("empty_output_retries", 0))
    return args


def base_record(row, args, gold, status, raw_output, raw_response, response_meta, server, started, error):
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
            "speaker_display": speaker_display(row),
            "text": row.get("text"),
            "conversation": conversation_text(row),
        },
        "raw_response_id": raw_response.get("id") if isinstance(raw_response, dict) else None,
        "response_meta": response_meta or {},
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    labels = load_labels(args.label_file)
    rows = list(read_jsonl(args.manifest))
    rows = [row for idx, row in enumerate(rows) if idx % args.num_shards == args.shard_index]
    if args.limit > 0:
        rows = rows[: args.limit]
    visual_by_id = load_reason_map(args.visual_reason_file, "visual") if args.step == "fusion" else {}
    dialogue_by_id = load_reason_map(args.dialogue_reason_file, "dialogue") if args.step == "fusion" else {}

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
                "visual_reason_files": args.visual_reason_file if args.step == "fusion" else None,
                "dialogue_reason_files": args.dialogue_reason_file if args.step == "fusion" else None,
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
        args.current_visual_reason = ""
        args.current_dialogue_reason = ""
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
        if args.step == "fusion":
            visual_row = visual_by_id.get(sample_id)
            dialogue_row = dialogue_by_id.get(sample_id)
            if not visual_row or not dialogue_row:
                counters["skipped_missing_reason"] += 1
                append_jsonl(
                    args.output,
                    {
                        "sample_id": sample_id,
                        "dataset": args.dataset,
                        "split": args.split,
                        "step": args.step,
                        "status": "skipped_missing_reason",
                        "gold": gold,
                        "has_visual_reason": bool(visual_row),
                        "has_dialogue_reason": bool(dialogue_row),
                    },
                )
                continue
            args.current_visual_reason = visual_row.get("visual_reason", "")
            args.current_dialogue_reason = dialogue_row.get("dialogue_reason", "")
        try:
            raw_output, raw_response, response_meta = generate_one(row, labels, args, server_for_shard)
            error = None
            if args.step == "visual":
                reason, status = extract_reason(raw_output, "VISUAL_REASON", strict_schema=True)
                record = base_record(
                    row, args, gold, status, raw_output, raw_response, response_meta, server_for_shard, started, error
                )
                record["visual_reason"] = reason
                counters["visual_reason_generated" if reason else "visual_reason_empty"] += 1
            elif args.step == "dialogue":
                reason, status = parse_dialogue_reason(raw_output)
                record = base_record(
                    row, args, gold, status, raw_output, raw_response, response_meta, server_for_shard, started, error
                )
                record["dialogue_reason"] = reason
                counters["dialogue_reason_generated" if reason else "dialogue_reason_parse_failed"] += 1
            elif args.step == "fusion":
                fusion_reason, final_answer, status, parse_error = parse_fusion_response(raw_output, gold, labels)
                record = base_record(
                    row, args, gold, status, raw_output, raw_response, response_meta, server_for_shard, started, error
                )
                record["fusion_reason"] = fusion_reason
                record["final_answer"] = final_answer
                record["parse_error"] = parse_error
                record["visual_reason"] = args.current_visual_reason
                record["dialogue_reason"] = args.current_dialogue_reason
                flags = count_fusion_flags(fusion_reason or "")
                record["fusion_flags"] = flags
                if status == "ok":
                    counters["fusion_generated_ok"] += 1
                    for flag_name, enabled in flags.items():
                        if enabled:
                            counters[flag_name] += 1
                else:
                    counters["fusion_parse_failed"] += 1
                    if parse_error:
                        counters[f"fusion_parse_failed_{parse_error}"] += 1
            else:
                prediction = extract_label(raw_output, labels)
                status = "ok" if prediction else "ok_unparsed"
                record = base_record(
                    row, args, gold, status, raw_output, raw_response, response_meta, server_for_shard, started, error
                )
                record["prediction"] = prediction
                record["correct"] = prediction == gold
                counters["prediction_generated" if prediction else "prediction_unparsed"] += 1
        except Exception as exc:
            counters["failed"] += 1
            record = base_record(row, args, gold, "failed", "", None, None, server_for_shard, started, repr(exc))

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
