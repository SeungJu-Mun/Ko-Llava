# config/env_config.py
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

def _env(name: str, default: Optional[str] = None, required: bool = False) -> str:
    val = os.getenv(name, default)
    if required and (val is None or val == ""):
        raise RuntimeError(f"Missing required env var: {name}")
    return val

@dataclass(frozen=True)
class EnvironmentConfig:
    LLAMA_MODEL_NAME: str
    LLAVA_MODEL_NAME: str
    CACHE_DIR: Path
    OUTPUT_DIR: Path

    @classmethod
    def load(cls, dotenv_path: Optional[str] = None) -> "EnvironmentConfig":
        load_dotenv(dotenv_path, override=False)

        cache_dir = Path(_env("CACHE_DIR", default="~/.cache/hf")).expanduser()
        out_dir   = Path(_env("OUTPUT_DIR", default="./outputs")).resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        return cls(
            LLAMA_MODEL_NAME=_env("LLAMA_MODEL_NAME", required=True),
            LLAVA_MODEL_NAME=_env("LLAVA_MODEL_NAME", required=True),
            CACHE_DIR=cache_dir,
            OUTPUT_DIR=out_dir,
        )
