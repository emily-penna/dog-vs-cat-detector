from ultralytics import YOLO

# Load modelt
model = YOLO('best.pt')

# Use the model
# results = model.train(data='coco128.yaml', epochs=3)  # train the model
# results = model.val()  # evaluate model performance on the validation set
model.predict(source='images', save=True)  # predict on an image
# results = model.export(format='onnx')  # export the model to ONNX format