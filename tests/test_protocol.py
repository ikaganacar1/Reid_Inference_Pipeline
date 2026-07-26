import json

import numpy as np
import pytest

from src.realtime.protocol import HEADER_STRUCT, MAGIC, decode_jpeg, encode_jpeg, pack_frame, unpack_frame


def test_packet_round_trip_preserves_capture_and_receive_times():
    detections = np.array([[3, 2, 12, 15, 0.9, 0]], dtype=np.float32)
    payload = pack_frame("cam1", 7, detections, b"frame-jpeg", [b"crop-jpeg"], 30, 20, 12.5)

    packet = unpack_frame(payload, received_at=12.7)

    assert packet.camera_id == "cam1"
    assert packet.frame_id == 7
    assert packet.timestamp == 12.5
    assert packet.received_at == 12.7
    np.testing.assert_allclose(packet.detections, detections)


def test_packet_rejects_crop_mismatch_and_invalid_header_length():
    detections = np.array([[0, 0, 5, 5, 0.9, 0]], dtype=np.float32)
    with pytest.raises(ValueError, match="Crop/detection"):
        pack_frame("cam1", 0, detections, b"frame-jpeg", [], 10, 10)

    malformed = HEADER_STRUCT.pack(MAGIC, 0)
    with pytest.raises(ValueError, match="header length"):
        unpack_frame(malformed)


def test_packet_rejects_unsafe_camera_id_and_malformed_detection_shape():
    with pytest.raises(ValueError, match="camera_id"):
        pack_frame("<script>", 0, np.empty((0, 6)), b"jpeg", [], 10, 10)

    header = {
        "camera_id": "cam1",
        "frame_id": 0,
        "timestamp": 1.0,
        "width": 10,
        "height": 10,
        "detections": [0, 0, 5, 5, 0.9, 0],
        "frame_size": 1,
        "crop_sizes": [],
    }
    header_bytes = json.dumps(header).encode()
    malformed = HEADER_STRUCT.pack(MAGIC, len(header_bytes)) + header_bytes + b"x"
    with pytest.raises(ValueError, match=r"shape \[N, 6\]"):
        unpack_frame(malformed)


def test_pillow_jpeg_codec_preserves_shape_and_channel_order():
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[:, :, 2] = 255
    decoded = decode_jpeg(encode_jpeg(image, 95))

    assert decoded.shape == image.shape
    assert decoded[:, :, 2].mean() > 240
    assert decoded[:, :, 0].mean() < 15
