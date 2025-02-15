"""
Author: Manjunath Mohan
Github: github.com/mnjm
"""
import numpy as np
import cv2

class ObjectBBox:
    """ Object Detection Bounding Box, used in YOLOv11 Wrapper """
    def __init__(self, label, conf, center_x, center_y, width, height, scale_x=1, scale_y=1):
        """
        Initialize BBox Object

        Args:
            label (str): Object label
            conf (float): Object confidence score
            center_x (float): Center of bbox in x axis
            center_y (float): Center of bbox in y axis
            width (flaot): Width of bbox
            height (float): Height of the bbox
            scale_x (float, optional): Scaling factor for x axis. Defaults to 1.
            scale_y (float, optional): Scaling factor for y axis. Defaults to 1.
        """
        self.label = label
        self.conf = conf
        scaled_center = center_x * scale_x, center_y * scale_y
        scaled_size = width * scale_x, height * scale_y
        half_size = scaled_size[0] * 0.5, scaled_size[1] * 0.5
        self.x1, self.y1 = scaled_center[0] - half_size[0], scaled_center[1] - half_size[1]
        self.x2, self.y2 = scaled_center[0] + half_size[0], scaled_center[1] + half_size[1]
        self.area = scaled_size[0] * scaled_size[1]

    def __str__(self):
        """
        String conversion for priniting

        Returns:
            str: String with bbox values
        """ 
        return f"{self.label} ({self.conf}) {self.x1, self.y1} {self.x2, self.y2}"

    def draw(self, image, bbox_color=(0,0,255), txt_color=(255,0,0)):
        """
        Draws bbox and label on an image

        Args:
            image (np.ndarray): Image
            bbox_color (tuple, optional): BGR color for bbox. Defaults to (0,0,255).
            txt_color (tuple, optional): BGR color for label. Defaults to (255,0,0).

        Returns:
            np.ndarray: Drawn image
        """
        label = self.label
        conf = self.conf
        x1, y1, x2, y2 = map(lambda x: int(x), (self.x1, self.y1, self.x2, self.y2))
        img_size = image.shape[:2]
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, int(max(img_size) * 0.0006))
        font_scale = int(max(img_size) * 0.0005)
        font_tickness = max(1, int(max(img_size) * 0.0006))
        cv2.putText(image, f"{label}({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, txt_color, font_tickness, cv2.LINE_AA)
        return image

def calc_iou(bbox1, bbox2):
    """
    Calculates IoU given 2 bounding box

    Args:
        bbox1 (ObjectBBox): First bbox
        bbox2 (ObjectBBos): Second bbox

    Returns:
        float: Calculated IoU
    """
    x1 = max(bbox1.x1, bbox2.x1)
    y1 = max(bbox1.y1, bbox2.y1)
    x2 = min(bbox1.x2, bbox2.x2)
    y2 = min(bbox1.y2, bbox2.y2)
    inter_w, inter_h = x2 - x1, y2 - y1
    if inter_w <= 0 or inter_h <= 0:
        return 0
    interArea = inter_w * inter_h
    union = bbox1.area + bbox2.area - interArea
    if union <= 0:
        return 0
    return interArea / union
