from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/last.pt")

model.export(format="onnx", simplify=True, device="cuda:0")
model.export(format="openvino", half=True)
model.export(format="openvino", int8=True, data="camera.yaml")
