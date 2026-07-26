import numpy as np
import pytest

from src.realtime.identity_gallery import IdentityGallery


def test_blocked_best_candidate_does_not_fall_through_to_second_best():
    gallery = IdentityGallery(match_threshold=0.3)
    first = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    second = np.array([0.99, 0.1, 0.0], dtype=np.float32)
    gid1 = gallery.create_new_identity("cam1", 1, first, timestamp=1.0)
    gid2 = gallery.create_new_identity("cam2", 1, second, timestamp=1.0)

    matched, info = gallery.preview_match(first, timestamp=1.1, blocked_global_ids={gid1})

    assert matched is None
    assert info["candidate_global_id"] == gid2
    assert info["blocked_candidate_global_id"] == gid1
    assert info["blocked_candidate_is_better"] is True


def test_gallery_rejects_nonfinite_embeddings_and_reset_restarts_ids():
    gallery = IdentityGallery()
    with pytest.raises(ValueError, match="non-finite"):
        gallery.create_new_identity("cam1", 1, np.array([np.nan, 0.0]))

    assert gallery.create_new_identity("cam1", 1, np.array([1.0, 0.0])) == 1
    gallery.reset()
    assert gallery.create_new_identity("cam1", 2, np.array([0.0, 1.0])) == 1


def test_delayed_observation_does_not_rewrite_newer_gallery_state():
    gallery = IdentityGallery(ema_alpha=0.5, max_exemplars=4)
    newer = np.array([1.0, 0.0], dtype=np.float32)
    delayed = np.array([0.0, 1.0], dtype=np.float32)
    gid = gallery.create_new_identity("cam-new", 1, newer, timestamp=10.0)

    gallery.assign_to_global("cam-old", 2, gid, delayed, timestamp=9.0)

    entry = gallery.entries[gid]
    np.testing.assert_allclose(entry.embedding, newer)
    assert len(entry.exemplars) == 1
    assert entry.last_seen == 10.0
    assert entry.camera_id == "cam-new"


def test_tracklet_prototypes_preserve_an_earlier_camera_appearance():
    gallery = IdentityGallery(
        match_threshold=0.3,
        ema_alpha=0.5,
        max_exemplars=4,
        use_exemplars_for_matching=True,
    )
    first_view = np.array([1.0, 0.0], dtype=np.float32)
    second_view = np.array([0.0, 1.0], dtype=np.float32)
    gid = gallery.create_new_identity("cam1", 1, first_view, timestamp=1.0)

    for timestamp in range(2, 20):
        gallery.assign_to_global("cam2", 2, gid, second_view, timestamp=float(timestamp))

    matched, info = gallery.preview_match(first_view, timestamp=20.0)

    assert matched == gid
    assert info["exemplar_distance"] == pytest.approx(0.0)
    assert len(gallery.entries[gid].exemplars) == 2


def test_bind_local_to_global_does_not_update_appearance():
    gallery = IdentityGallery(match_threshold=0.3)
    original = np.array([1.0, 0.0], dtype=np.float32)
    gid = gallery.create_new_identity("cam1", 1, original, timestamp=1.0)
    before = gallery.entries[gid].embedding.copy()

    gallery.bind_local_to_global("cam2", 2, gid)

    assert gallery.get_existing_global_id("cam2", 2) == gid
    np.testing.assert_allclose(gallery.entries[gid].embedding, before)
    assert gallery.entries[gid].last_seen == 1.0
