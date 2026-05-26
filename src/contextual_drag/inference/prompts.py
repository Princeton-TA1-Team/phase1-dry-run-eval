"""Template loading, prompt rendering, prompt-hash resume key, and JSONL scan.

The resume key is sha256(rendered_prompt)[:16]. Stable across reruns because
all inputs (row content, template, chat template, enable_thinking) are
deterministic. Handles the 1F-flattened-from-2F dataset case where the same
`id` appears in two rows with different drafts — the hash differs because
the rendered prompt differs.
"""

import hashlib
import json
import re
from pathlib import Path

_TEMPLATE_FIELD_RE = re.compile(r"\{arg_([^}]+)\}")


def load_template(template_path: Path, template_key: str) -> str:
    with open(template_path) as f:
        templates = json.load(f)
    if template_key not in templates:
        raise SystemExit(
            f"template key {template_key!r} not found in {template_path}. "
            f"Available: {sorted(templates)}"
        )
    return templates[template_key]


def template_fields(template: str) -> list[str]:
    """Return sorted list of arg_ field names referenced by the template."""
    return sorted(set(_TEMPLATE_FIELD_RE.findall(template)))


def render_prompt(template: str, row: dict, tokenizer, enable_thinking: bool) -> str:
    """Substitute {arg_*} placeholders with row values, then apply the
    tokenizer's chat template (with `add_generation_prompt=True`)."""
    fields = set(_TEMPLATE_FIELD_RE.findall(template))
    user_content = template
    for f in fields:
        if f not in row:
            raise KeyError(
                f"template field 'arg_{f}' missing from row "
                f"(have: {sorted(row)})"
            )
        user_content = user_content.replace(f"{{arg_{f}}}", str(row[f]))
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def prompt_hash(prompt: str) -> str:
    """16-char prefix of sha256(prompt). Used as the per-row resume key."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def load_completed_hashes(jsonl_path: Path, task_name: str) -> set[str]:
    """Scan an on-disk JSONL and return the set of completed prompt hashes,
    extracted from `<task_name>_generations_metadata.prompt_hash`."""
    if not jsonl_path.exists():
        return set()
    meta_key = f"{task_name}_generations_metadata"
    done: set[str] = set()
    bad = 0
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            h = (obj.get(meta_key, {}) or {}).get("prompt_hash")
            if h:
                done.add(h)
    if bad:
        print(f"  [warn] skipped {bad} malformed JSONL lines in {jsonl_path}",
              flush=True)
    return done
