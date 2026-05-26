from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any


def resource_path(package: str, name: str) -> Path:
    return Path(resources.files(package).joinpath(name))


def load_json_resource(package: str, name: str) -> Any:
    with resources.files(package).joinpath(name).open("r", encoding="utf-8") as file:
        return json.load(file)


def inference_model_config_resource() -> Any:
    return load_json_resource("contextual_drag.resources.inference", "eval_models_params.json")


def crux_dataset_resource_path() -> Path:
    return resource_path("contextual_drag.resources.evaluation.crux", "cruxeval.jsonl")
