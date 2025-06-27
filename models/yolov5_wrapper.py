import torch
import numpy as np
import cv2
from PIL import Image
from models.yolov3.image_utils import letterbox_image
import os
import logging
from ultralytics import YOLO

class YOLOv3(object):
    _defaults = {
        "box_score_threshold": 0.3,
        "nms_iou_threshold": 0.45,
        "mAP_iou_threshold": 0.5,
        "model_image_size": (416, 416),
        "gpu_num": 1,
    }
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name'" + n + "'"
        
    def __init__(self, device=None, **kwargs):
        self.__dict__.update(self._defaults)
        self.__dict__.update(kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Loading yolov5 from torch.hub...")
        # self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(self.device).eval()
        self.model = YOLO('yolov5su.pt').to(self.device).eval()
        self.class_names = self.model.names
        self.model_image_size = (416, 416) 
        self.logger.info("Model loaded successfully.")

    def predict(self, image, show_image=False):
        """
        Returns:
            - pred_dict: {
                'boxes': [[top, left, bottom, right], ...],
                'scores': [float, ...],
                'classes': [int, ...]
            }
            - raw_tensor: shape [num_boxes, 6+], differentiable output
        """
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        if isinstance(image, np.ndarray) and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image

        image_tensor = torch.from_numpy(image_bgr).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0

        with torch.no_grad():
            results = self.model(image_tensor, imgsz=self.model_image_size[0])
        
        output = results[0]  # [num_boxes, 6+] with grads if no torch.no_grad()
        boxes_xyxy = output.boxes.xyxy  # [N, 4]
        scores = output.boxes.conf
        classes = output.boxes.cls

        pred_dict = {'boxes': [], 'scores': [], 'classes': []}
        for box, conf, cls_idx in zip(boxes_xyxy, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            if conf < self.box_score_threshold:
                continue
            pred_dict['boxes'].append([y1, x1, y2, x2])  # HWC format as before
            pred_dict['scores'].append(float(conf))
            pred_dict['classes'].append(int(cls_idx))

        return pred_dict, output
        

    def forward_raw(self, image_tensor):
        """
        Run raw forward pass through model to get differentiable outputs.
        Assumes image_tensor is preprocessed: shape [1, 3, H, W] and normalized.
        """
        return self.model.model(image_tensor)

    def forward_raw_logits(self, image_tensor):
        """
        Manually forward through YOLOv5 to get raw outputs from Detect head, preserving gradients.
        """
        self.model.eval()
        with torch.set_grad_enabled(True):
            x = image_tensor
            features = []
            layers = list(self.model.model.children())

            for m in layers[:-1]:  # all except final Detect
                x = m(x)
                if isinstance(x, (list, tuple)):
                    # Happens with some FPN layers; collect all feature maps
                    features.extend(x)
                else:
                    features.append(x)

            detect_layer = layers[-1]
            output = detect_layer(features[-len(detect_layer.stride):])  # pass last N feature maps
            return output





    # def predict(self, image, show_image=False):
    #     """
    #     Input:
    #         image: PIL.Image.Image or np.ndarray
    #     Output:
    #         {
    #             'boxes': [[top, left, bottom, right], ...],
    #             'scores': [float, ...],
    #             'classes': [int, ...]
    #         }
    #     """
    #     # Convert PIL to OpenCV format (BGR)
    #     if isinstance(image, Image.Image):
    #         image = np.array(image.convert("RGB"))
    #     if isinstance(image, np.ndarray) and image.shape[2] == 3:
    #         image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #     else:
    #         image_bgr = image  # assume already BGR

    #     # Run inference
    #     # results = self.model.predict(image_bgr, imgsz=self.model_image_size[0])
    #     results = self.model(image_bgr, imgsz=self.model_image_size[0])
    #     output = results[0]
    #     boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
    #     scores = results[0].boxes.conf.cpu().numpy()
    #     classes = results[0].boxes.cls.cpu().numpy()

    #     pred_dict = {'boxes': [], 'scores': [], 'classes': []}
    #     for box, conf, cls_idx in zip(boxes_xyxy, scores, classes):
    #         x1, y1, x2, y2 = map(int, box)
    #         if conf < self.box_score_threshold:
    #             continue
    #         # ⚠️ Match your pipeline’s [top, left, bottom, right] format
    #         pred_dict['boxes'].append([y1, x1, y2, x2])
    #         pred_dict['scores'].append(float(conf))
    #         pred_dict['classes'].append(int(cls_idx))

    #         if show_image:
    #             label = f"{self.class_names[int(cls_idx)]} {conf:.2f}"
    #             cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #             cv2.putText(image_bgr, label, (x1, y1 - 10),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    #     if show_image:
    #         cv2.imshow("Prediction", image_bgr)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()

    #     return pred_dict
