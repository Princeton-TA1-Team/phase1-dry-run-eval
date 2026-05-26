from __future__ import annotations

import json
from pathlib import Path

from contextual_drag.config.paths import resolve_split_file


def load_split_ids(
    split: str,
    split_root: str | Path | None = None,
    execution_mode: str | None = None,
) -> list[str]:
    split_ids: list[str] = []
    for split_file in resolve_split_file(split, split_root=split_root, execution_mode=execution_mode):
        with Path(split_file).open("r", encoding="utf-8") as file:
            split_ids.extend(json.load(file))
    return split_ids
