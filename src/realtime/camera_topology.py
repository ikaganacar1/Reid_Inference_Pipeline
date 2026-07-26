"""Camera-overlap policy shared by realtime assignment and offline audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _parse_camera_pairs(value: Any, pair_type: str) -> frozenset[frozenset[str]]:
    if value is None or value == "":
        return frozenset()
    if isinstance(value, str):
        raw_pairs: list[Any] = [item.strip() for item in value.split(",") if item.strip()]
    else:
        raw_pairs = list(value)

    pairs: set[frozenset[str]] = set()
    for raw_pair in raw_pairs:
        if isinstance(raw_pair, str):
            cameras = [item.strip() for item in raw_pair.split(":")]
        else:
            cameras = [str(item).strip() for item in raw_pair]
        if len(cameras) != 2 or not all(cameras) or cameras[0] == cameras[1]:
            raise ValueError(
                f"Each {pair_type} camera pair must contain two different camera IDs; "
                f"got {raw_pair!r}"
            )
        pairs.add(frozenset(cameras))
    return frozenset(pairs)


def parse_overlapping_camera_pairs(value: Any) -> frozenset[frozenset[str]]:
    """Normalize overlapping camera pairs from YAML or CLI syntax."""
    return _parse_camera_pairs(value, "overlapping")


def parse_adjacent_camera_pairs(value: Any) -> frozenset[frozenset[str]]:
    """Normalize fast-handoff camera pairs from YAML or CLI syntax."""
    return _parse_camera_pairs(value, "adjacent")


@dataclass(frozen=True)
class CameraTopology:
    """Describe which camera pairs may contain one person simultaneously."""

    allow_all_overlaps: bool = False
    overlapping_pairs: frozenset[frozenset[str]] = frozenset()
    adjacent_pairs: frozenset[frozenset[str]] = frozenset()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CameraTopology":
        return cls(
            allow_all_overlaps=bool(config.get("allow_all_camera_overlap", False)),
            overlapping_pairs=parse_overlapping_camera_pairs(
                config.get("overlapping_camera_pairs", [])
            ),
            adjacent_pairs=parse_adjacent_camera_pairs(
                config.get("adjacent_camera_pairs", [])
            ),
        )

    def may_overlap(self, first_camera: str, second_camera: str) -> bool:
        if first_camera == second_camera:
            return True
        return self.allow_all_overlaps or frozenset((first_camera, second_camera)) in self.overlapping_pairs

    def as_pairs(self) -> list[list[str]]:
        return sorted(sorted(pair) for pair in self.overlapping_pairs)

    def are_adjacent(self, first_camera: str, second_camera: str) -> bool:
        if first_camera == second_camera:
            return False
        return frozenset((first_camera, second_camera)) in self.adjacent_pairs

    def as_adjacent_pairs(self) -> list[list[str]]:
        return sorted(sorted(pair) for pair in self.adjacent_pairs)
