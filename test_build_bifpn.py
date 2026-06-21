"""Quick build check for yolov8s-p2-bifpn-lite."""
import ultralytics_rainforest  # noqa: F401
from ultralytics import YOLO

model = YOLO("cfg/yolov8s-p2-bifpn-lite.yaml")
model.info()
d = model.model.model[-1]
print(f"Detect f={d.f} n_layers={len(model.model.model)}")
print("✅ yolov8s-p2-bifpn-lite build ok")
