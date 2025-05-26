import torch
import cv2
import numpy as np
from models.yolov5_wrapper import YOLOv3
from PIL import Image

img = Image.open('output/ori_3.png')
detector = YOLOv3()
results = detector.predict(img, show_image=True)
print(results)

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# imgs = ['output/ori_3.png']
# img_bgr = cv2.imread(imgs[0])
# # cv2img = cv2.cvtColor(cv2img, COLOR_BGR2RGB)

# results = model(imgs)
# detections = results.xyxy[0]

# # Draw boxes
# for *box, conf, cls in detections:
#     x1, y1, x2, y2 = map(int, box)
#     label = f"{model.names[int(cls)]} {conf:.2f}"
#     cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     cv2.putText(img_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# # Show image
# cv2.imshow("Detections", img_bgr)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

