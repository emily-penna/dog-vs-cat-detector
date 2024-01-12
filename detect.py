from ultralytics import YOLO

# Load model
model = YOLO('best.pt')

# Use the model
model.predict(source='images', save=True)