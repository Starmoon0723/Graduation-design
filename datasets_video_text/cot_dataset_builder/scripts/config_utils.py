#!/usr/bin/env python3
import json
from pathlib import Path


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "qwen36_27b_dataset.yaml"
)


def load_yaml(path):
    try:
        import yaml
    except ImportError:
        return load_simple_yaml(path)
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_scalar(value):
    value = value.strip()
    if value == "":
        return ""
    if value in ("true", "True"):
        return True
    if value in ("false", "False"):
        return False
    if value in ("null", "None", "~"):
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(part.strip()) for part in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def yaml_lines(path):
    lines = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        lines.append((indent, raw.strip()))
    return lines


def parse_key_value(text):
    if ":" not in text:
        raise ValueError(f"Unsupported YAML line: {text}")
    key, value = text.split(":", 1)
    return key.strip(), value.strip()


def parse_block(lines, index, indent):
    if index >= len(lines):
        return {}, index
    is_list = lines[index][0] == indent and lines[index][1].startswith("- ")
    if is_list:
        result = []
        while index < len(lines):
            line_indent, text = lines[index]
            if line_indent < indent:
                break
            if line_indent != indent or not text.startswith("- "):
                break
            item_text = text[2:].strip()
            index += 1
            if not item_text:
                item, index = parse_block(lines, index, indent + 2)
                result.append(item)
                continue
            if ":" in item_text:
                key, value = parse_key_value(item_text)
                item = {}
                if value:
                    item[key] = parse_scalar(value)
                else:
                    item[key], index = parse_block(lines, index, indent + 2)
                while index < len(lines) and lines[index][0] > indent:
                    child_indent, child_text = lines[index]
                    if child_indent != indent + 2 or child_text.startswith("- "):
                        break
                    child_key, child_value = parse_key_value(child_text)
                    index += 1
                    if child_value:
                        item[child_key] = parse_scalar(child_value)
                    else:
                        item[child_key], index = parse_block(lines, index, child_indent + 2)
                result.append(item)
            else:
                result.append(parse_scalar(item_text))
        return result, index

    result = {}
    while index < len(lines):
        line_indent, text = lines[index]
        if line_indent < indent:
            break
        if line_indent != indent:
            break
        key, value = parse_key_value(text)
        index += 1
        if value:
            result[key] = parse_scalar(value)
        else:
            result[key], index = parse_block(lines, index, indent + 2)
    return result, index


def load_simple_yaml(path):
    lines = yaml_lines(path)
    data, index = parse_block(lines, 0, 0)
    if index != len(lines):
        raise ValueError(f"Unsupported YAML structure near line: {lines[index]}")
    return data


def load_config(path=None):
    config_path = Path(path or DEFAULT_CONFIG)
    cfg = load_yaml(config_path)
    cfg["_config_path"] = str(config_path)
    return cfg


def project_root(cfg):
    return Path(cfg["project"]["root"])


def resolve_project_path(cfg, value):
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root(cfg) / path


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def json_dumps_compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
