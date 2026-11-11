from ultralytics import YOLO

# Load YOLOv8 pretrained model
model = YOLO("yolov8n.pt")

# Train the model
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=224,
    batch=8
)

print("Training completed!")