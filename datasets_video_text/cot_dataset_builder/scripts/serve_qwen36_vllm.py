#!/usr/bin/env python3
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from config_utils import json_dumps_compact, load_config, resolve_project_path


def pid_file(cfg):
    return resolve_project_path(cfg, cfg["output"]["logs"]) / "vllm_servers.pid.json"


def log_dir(cfg):
    path = resolve_project_path(cfg, cfg["output"]["logs"]) / "serve"
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_running(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def build_command(cfg, instance):
    model = cfg["model"]
    vllm = cfg["vllm"]
    cmd = [
        "vllm",
        "serve",
        model["path"],
        "--served-model-name",
        model.get("served_model_name", model["name"]),
        "--host",
        str(vllm.get("host", "127.0.0.1")),
        "--port",
        str(instance["port"]),
        "--tensor-parallel-size",
        str(instance["tensor_parallel_size"]),
        "--max-model-len",
        str(vllm["max_model_len"]),
        "--gpu-memory-utilization",
        str(vllm.get("gpu_memory_utilization", 0.9)),
        "--media-io-kwargs",
        json_dumps_compact(vllm["media_io_kwargs"]),
    ]
    if vllm.get("enable_reasoning", True):
        cmd.append("--enable-reasoning")
    if vllm.get("reasoning_parser"):
        cmd.extend(["--reasoning-parser", str(vllm["reasoning_parser"])])
    if model.get("trust_remote_code", True):
        cmd.append("--trust-remote-code")
    allowed_path = vllm.get("allowed_local_media_path")
    if allowed_path:
        cmd.extend(["--allowed-local-media-path", str(allowed_path)])
    return cmd


def load_pid_records(path):
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def start(cfg):
    records = load_pid_records(pid_file(cfg))
    alive = [rec for rec in records if is_running(rec.get("pid"))]
    if alive:
        print(json.dumps({"event": "already_running", "servers": alive}, ensure_ascii=False, indent=2))
        return

    env_file = cfg["project"].get("env_file")
    logs = log_dir(cfg)
    new_records = []
    for instance in cfg["vllm"]["instances"]:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(instance["cuda_visible_devices"])
        log_path = logs / f'{instance["name"]}.log'
        cmd = build_command(cfg, instance)
        with log_path.open("ab") as log_f:
            log_f.write(
                (
                    f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} starting {instance['name']} =====\n"
                    f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n"
                    f"ENV_FILE={env_file or ''}\n"
                    f"CMD={' '.join(cmd)}\n"
                ).encode("utf-8")
            )
            proc = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        record = {
            "name": instance["name"],
            "pid": proc.pid,
            "base_url": f"http://{cfg['vllm'].get('host', '127.0.0.1')}:{instance['port']}/v1",
            "cuda_visible_devices": instance["cuda_visible_devices"],
            "log": str(log_path),
            "cmd": cmd,
        }
        new_records.append(record)
        print(json.dumps({"event": "started", **record}, ensure_ascii=False))

    pid_path = pid_file(cfg)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(json.dumps(new_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stop(cfg):
    records = load_pid_records(pid_file(cfg))
    for rec in records:
        pid = rec.get("pid")
        if not pid or not is_running(pid):
            print(json.dumps({"event": "not_running", "name": rec.get("name"), "pid": pid}, ensure_ascii=False))
            continue
        try:
            os.killpg(os.getpgid(int(pid)), signal.SIGTERM)
        except Exception:
            os.kill(int(pid), signal.SIGTERM)
        print(json.dumps({"event": "sent_sigterm", "name": rec.get("name"), "pid": pid}, ensure_ascii=False))

    time.sleep(3)
    for rec in records:
        pid = rec.get("pid")
        if pid and is_running(pid):
            try:
                os.killpg(os.getpgid(int(pid)), signal.SIGKILL)
            except Exception:
                os.kill(int(pid), signal.SIGKILL)
            print(json.dumps({"event": "sent_sigkill", "name": rec.get("name"), "pid": pid}, ensure_ascii=False))


def status(cfg):
    records = load_pid_records(pid_file(cfg))
    for rec in records:
        rec = dict(rec)
        rec["running"] = bool(rec.get("pid") and is_running(rec["pid"]))
        print(json.dumps(rec, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "stop", "status"])
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.action == "start":
        start(cfg)
    elif args.action == "stop":
        stop(cfg)
    else:
        status(cfg)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
