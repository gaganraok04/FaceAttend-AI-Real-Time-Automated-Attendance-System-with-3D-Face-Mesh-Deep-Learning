from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Test the model on the test dataset
results = model.val(
    data="data.yaml",
    split="test",
    imgsz=224
)

# Print test results
print("\n========== TEST RESULTS ==========")
print("Precision :", results.box.mp)
print("Recall    :", results.box.mr)
print("mAP50     :", results.box.map50)
print("mAP50-95  :", results.box.map)
print("==================================")