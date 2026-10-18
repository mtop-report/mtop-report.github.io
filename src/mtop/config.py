"""Load the YAML configuration (settings + predictor registry)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import yaml

from .paths import CONFIG


@dataclass(frozen=True)
class Feature:
    key: str
    stage: str
    type: str
    monotone: int
    spec: dict

    @property
    def columns(self) -> list[str]:
        """Model columns produced by this predictor (one-hot for categoricals). An option marked
        `model: false` is offered for entry but has no model term yet (too few patients)."""
        if self.type == "categorical":
            ref = self.spec.get("reference")
            return [f"{self.key}__{o['value']}" for o in self.spec["options"]
                    if o["value"] != ref and o.get("model", True)]
        return [self.key]


@lru_cache(maxsize=None)
def settings() -> dict:
    with open(CONFIG / "settings.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=None)
def features() -> tuple[Feature, ...]:
    with open(CONFIG / "features.yaml", encoding="utf-8") as f:
        raw = yaml.safe_load(f)["features"]
    return tuple(Feature(r["key"], r["stage"], r["type"], int(r.get("monotone", 0)), r) for r in raw)


def stage_ids() -> list[str]:
    return [s["id"] for s in sorted(settings()["stages"], key=lambda s: s["order"])]


def stage_spec(stage: str) -> dict:
    return next(s for s in settings()["stages"] if s["id"] == stage)


def stage_features(stage: str) -> list[Feature]:
    order = {s: i for i, s in enumerate(stage_ids())}
    return [f for f in features() if order[f.stage] <= order[stage]]


def targets() -> dict:
    return settings()["targets"]


def primary_target() -> str:
    return next(k for k, v in targets().items() if v.get("primary"))
