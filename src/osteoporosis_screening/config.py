from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    raw: Path
    interim: Path
    processed: Path
    artifacts: Path
    figures: Path
    tables: Path

    @classmethod
    def from_root(cls, root: Path) -> "ProjectPaths":
        root = root.resolve()
        return cls(
            root=root,
            raw=root / "data" / "raw_xpt",
            interim=root / "data" / "interim",
            processed=root / "data" / "processed",
            artifacts=root / "artifacts",
            figures=root / "reports" / "figures",
            tables=root / "reports" / "tables",
        )

    def ensure(self) -> None:
        for path in (
            self.raw,
            self.interim,
            self.processed,
            self.artifacts,
            self.figures,
            self.tables,
        ):
            path.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)

