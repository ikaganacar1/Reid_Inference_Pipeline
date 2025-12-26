# Models Directory

Place your model files here:

## YOLO Model
- Download YOLO11n: `yolo11n.pt`
- Or use any Ultralytics YOLO model

## TAO ReID Model
- ONNX model: `lttc_0.1.4.49.onnx`
- Or your custom TAO ReID model in ONNX format

**Note:** Model files are excluded from git via `.gitignore` due to large size.

### Download Links
- YOLO: https://github.com/ultralytics/ultralytics
- TAO ReID: Contact your NVIDIA TAO provider or train your own model

### Model Structure Expected
```
models/
├── yolo11n.pt           # YOLO detection model
├── lttc_0.1.4.49.onnx   # TAO ReID ONNX model
└── README.md            # This file
```
