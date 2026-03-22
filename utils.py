import json
from pathlib import Path
from typing import Any


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def print_section(title: str) -> None:
    print(f"\n{'=' * 20} {title} {'=' * 20}")
