import pytest

from src.realtime.camera_topology import (
    CameraTopology,
    parse_adjacent_camera_pairs,
    parse_overlapping_camera_pairs,
)


def test_camera_topology_supports_pairs_and_all_overlap():
    topology = CameraTopology.from_config(
        {
            "overlapping_camera_pairs": [["cam1", "cam2"], "cam2:cam3"],
            "adjacent_camera_pairs": ["cam3:cam4"],
        }
    )

    assert topology.may_overlap("cam1", "cam2")
    assert topology.may_overlap("cam3", "cam2")
    assert not topology.may_overlap("cam1", "cam3")
    assert topology.are_adjacent("cam4", "cam3")
    assert not topology.may_overlap("cam3", "cam4")
    assert CameraTopology(allow_all_overlaps=True).may_overlap("cam1", "cam9")


def test_camera_topology_rejects_self_or_malformed_pairs():
    with pytest.raises(ValueError):
        parse_overlapping_camera_pairs("cam1:cam1")
    with pytest.raises(ValueError):
        parse_overlapping_camera_pairs("cam1")
    with pytest.raises(ValueError):
        parse_adjacent_camera_pairs("cam2:cam2")
