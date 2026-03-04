import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def load_config() -> dict:
    config_path = Path(__file__).resolve().parent / "config.yaml"
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing config file: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid config format in {config_path}. Use JSON-compatible YAML syntax."
        ) from exc
