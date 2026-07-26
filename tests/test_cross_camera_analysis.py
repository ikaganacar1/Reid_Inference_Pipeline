from scripts.analyze_cross_camera_reid import analyze_events


def make_event(
    timestamp,
    camera_id,
    local_track_id,
    global_id,
    reason="existing_local_track_verified",
    **match_fields,
):
    return {
        "time": timestamp,
        "camera_id": camera_id,
        "frame_id": int(timestamp * 10),
        "local_track_id": local_track_id,
        "global_id": global_id,
        "match": {"reason": reason, **match_fields},
    }


def test_analyze_events_reports_journeys_conflicts_recoveries_and_rejections():
    events = [
        make_event(0.0, "cam-a", 1, 1),
        make_event(0.1, "cam-a", 1, 1),
        make_event(
            3.0,
            "cam-b",
            2,
            1,
            "appearance_match_verified",
            best_distance=0.20,
            second_distance=0.40,
        ),
        make_event(3.1, "cam-b", 2, 1),
        make_event(10.0, "cam-a", 3, 2),
        make_event(10.0, "cam-b", 4, 2),
        make_event(20.0, "cam-c", 5, 3),
        make_event(20.0, "cam-c", 6, 3),
        make_event(30.0, "cam-d", 7, 4, remap_suppressed_reason="edge_partial"),
        make_event(32.0, "cam-d", 8, 4, "appearance_match_verified", best_distance=0.15),
        make_event(
            40.0,
            "cam-e",
            9,
            5,
            "appearance_remap",
            previous_global_id=6,
            candidate_global_id=5,
            best_distance=0.18,
        ),
        make_event(
            50.0,
            "cam-f",
            10,
            7,
            "new_identity",
            candidate_global_id=1,
            best_distance=0.24,
            second_distance=0.29,
        ),
    ]

    report = analyze_events(events, min_travel_seconds=2.0)

    assert report["summary"]["cross_camera_identity_count"] == 2
    assert report["summary"]["cross_camera_transition_count"] == 2
    assert report["summary"]["fast_or_overlapping_transition_count"] == 1
    assert report["summary"]["simultaneous_cross_camera_conflict_count"] == 1
    assert report["summary"]["same_camera_duplicate_count"] == 1
    assert report["summary"]["same_camera_local_recovery_count"] == 1
    assert report["summary"]["appearance_remap_count"] == 1
    assert report["summary"]["remap_suppression_count"] == 1
    assert report["summary"]["close_rejected_new_identity_count"] == 1
    assert report["close_rejected_new_identities"][0]["rejection_cause"] == "ambiguity_margin"

    journey = next(item for item in report["journeys"] if item["global_id"] == 1)
    assert journey["transitions"][0]["from_camera"] == "cam-a"
    assert journey["transitions"][0]["to_camera"] == "cam-b"
    assert journey["transitions"][0]["gap_seconds"] == 2.9


def test_analyzer_classifies_configured_simultaneous_overlap_as_allowed():
    events = [
        make_event(10.0, "cam-a", 1, 1),
        make_event(10.0, "cam-b", 2, 1),
    ]

    report = analyze_events(events, overlapping_camera_pairs="cam-a:cam-b")

    assert report["summary"]["simultaneous_cross_camera_conflict_count"] == 0
    assert report["summary"]["simultaneous_allowed_overlap_count"] == 1
    assert report["simultaneous_allowed_overlaps"][0]["overlapping_camera_pairs"] == [
        ["cam-a", "cam-b"]
    ]


def test_analyzer_classifies_fast_adjacent_handoff_as_allowed():
    events = [
        make_event(1.0, "cam-a", 1, 9),
        make_event(2.0, "cam-b", 2, 9, "appearance_match_verified_adjacent_handoff"),
    ]

    report = analyze_events(
        events,
        min_travel_seconds=2.0,
        adjacent_camera_pairs="cam-a:cam-b",
    )

    assert report["summary"]["allowed_adjacent_transition_count"] == 1
    assert report["summary"]["fast_nonoverlap_transition_count"] == 0
