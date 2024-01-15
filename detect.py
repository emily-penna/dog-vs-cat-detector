from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load model
model = YOLO('best.pt')
# Use the model
# model.predict(source='cat.jpg')
results = model('images', save=True, conf=0.5)
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    print(boxes)

# image_input = cv2.imread('cat.jpg')

# plt.rcParams['figure.figsize'] = (10.0, 10.0)
# plt.imshow(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
# plt.show()