import importlib.util

import pytest
import torch


if importlib.util.find_spec("boxmot") is None:
    pytest.skip("BoxMOT is not installed", allow_module_level=True)

from src.tracker import ExternalReIDBotSort, _BoxMOTBaseTrack


def test_constructing_multiple_trackers_does_not_reset_shared_id_counter():
    kwargs = {
        "reid_weights": None,
        "device": torch.device("cpu"),
        "half": False,
        "track_buffer": 30,
        "cmc_method": "sof",
    }
    _BoxMOTBaseTrack._count = 41
    ExternalReIDBotSort(with_reid=False, **kwargs)
    assert _BoxMOTBaseTrack._count == 41

    _BoxMOTBaseTrack._count = 47
    ExternalReIDBotSort(with_reid=False, **kwargs)
    assert _BoxMOTBaseTrack._count == 47
