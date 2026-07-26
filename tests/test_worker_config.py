import yaml

import pytest

from src.realtime.worker import RealtimeWorker, load_worker_config


def test_prime_url_environment_override(monkeypatch, tmp_path):
    config_path = tmp_path / "realtime.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "network": {"prime_url": "ws://old:8765"},
                "worker": {"camera_id": "cam1", "source": 0},
            }
        )
    )
    monkeypatch.setenv("PRIME_URL", "ws://new:8765")

    config = load_worker_config(config_path)

    assert config["network"]["prime_url"] == "ws://new:8765"


def test_send_deadline_keeps_fixed_rate_after_small_and_large_delays():
    deadline = RealtimeWorker.advance_send_deadline(None, now=10.03, interval=0.1)
    assert deadline == pytest.approx(10.13)

    deadline = RealtimeWorker.advance_send_deadline(deadline, now=10.14, interval=0.1)
    assert deadline == pytest.approx(10.23)

    deadline = RealtimeWorker.advance_send_deadline(deadline, now=10.56, interval=0.1)
    assert deadline == pytest.approx(10.63)
