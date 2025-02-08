"""
YOLOv11 ONNX Inferenence wrapper

Author: Manjunath Mohan
Github: github.com/mnjm
"""

import numpy as np
import onnxruntime as ort
from ast import literal_eval
import cv2
from bbox import ObjectBBox, calc_iou
from pathlib import Path

class YOLOv11:
    """ Wrapper for YOLOv11 ONNX Model (uses ONNXRuntime to run the model) """
    providers = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]

    def __init__(self, model_path, min_conf=0.35, iou_thresh=0.45, valid_class_checker=None):
        """
        Initialize YOLOv11 model for object detection.

        Args:
            model_path (str): Path to the ONNX model file.
            min_conf (float): Minimum confidence threshold for detections.
            iou_thresh (float): Intersection over Union (IoU) threshold for Non-Max Suppression.
            valid_class_checker (callable, optional): Function to filter valid class indices. Defaults to all classes.
        """
        self.model_path = model_path
        self.ort_sess = ort.InferenceSession(
            model_path, providers=YOLOv11.providers
        )
        self.input_name = self.ort_sess.get_inputs()[0].name
        self.output_name = self.ort_sess.get_outputs()[0].name
        self.min_conf = min_conf
        self.iou_thresh = iou_thresh
        self.valid_class_checker = valid_class_checker if valid_class_checker else lambda cls: True
        
        meta = self.ort_sess.get_modelmeta()
        
        ## Get Model Input Size
        custom_metadata = meta.custom_metadata_map
        assert "imgsz" in custom_metadata, "Error: ONNX model does not contain 'imgsz' metadata"
        self.input_size = literal_eval(custom_metadata['imgsz'])
        self.input_size = np.array(self.input_size)
        
        ## Get class name map from onnx model's metadata
        assert "names" in custom_metadata, "Error: ONNX model does not contain 'names' metadata"
        self.class_name_map = literal_eval(custom_metadata["names"])

    def _preprocess_input(self, input_img):
        """
        Preprocess the input image for YOLOv11.

        Args:
            input_img (np.ndarray): Original image in HxWxC format.

        Returns:
            np.ndarray: Preprocessed image in CxHxW format, normalized and resized.
        """
        inp = cv2.resize(input_img, self.input_size)
        inp = inp.astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1)) # from HxWxC to CxHxW
        inp = np.expand_dims(inp, axis=0)
        return inp

    def _postprocess_output(self, raw_out, original_shape):
        """
        Perform Non-Maximum Suppression (NMS) on model output.

        Args:
            raw_out (np.ndarray): Raw output from the model.
            original_shape (tuple): Original image shape (height, width).

        Returns:
            list: List of detected ObjectBBox objects.
        """
        raw_out = np.squeeze(raw_out)
        assert raw_out.shape[0] == 4 + len(self.class_name_map), f"Output not in valid shape {raw_out.shape}"
        n_det = raw_out.shape[1]
        assert n_det > 0, f"Output not in valid shape {raw_out.shape}"
        scale = np.array(original_shape) / self.input_size

        # List to keep track of valid bboxs
        valid_bbox_l = []
        # Keep track of suppressed detections
        suppresed_mask = np.zeros(n_det, dtype=bool)

        ## Sort object scores in descending order
        # find out the max scored class in each detection
        class_idx_l = np.argmax(raw_out[4:, :], axis=0)
        # get max scores for each detection
        max_conf_l = np.take_along_axis(raw_out[4:, :], class_idx_l[None, :], axis=0).squeeze()
        # now sort based on scores
        order_idx_l = np.argsort(max_conf_l)[::-1] # descending

        ## Iterate through all detections
        for i, idx1 in enumerate(order_idx_l):
            cls = class_idx_l[idx1]
            conf = max_conf_l[idx1]
            # if it is suppressed or if not a valid class or if conf is less then min_conf
            if suppresed_mask[idx1] or (not self.valid_class_checker(cls)) or conf < self.min_conf:
                continue
            lbl = self.class_name_map[cls]
            bbox1 = ObjectBBox(lbl, conf, *raw_out[:4, idx1], scale[1], scale[0])
            ## Select detection as valid
            valid_bbox_l.append(bbox1)

            for idx2 in order_idx_l[i+1:]:
                cls = class_idx_l[idx2]
                conf = max_conf_l[idx2]
                # if it is suppressed or if not a valid class or if conf is less then min_conf
                if suppresed_mask[idx2] or (not self.valid_class_checker(cls)) or conf < self.min_conf:
                    continue
                lbl = self.class_name_map[cls]
                bbox2 = ObjectBBox(lbl, conf, *raw_out[:4, idx2], scale[1], scale[0])
                iou = calc_iou(bbox1, bbox2)
                if (iou > self.iou_thresh):
                    suppresed_mask[idx2] = True

        return valid_bbox_l

    def detect(self, image):
        """
        Run object detection on an image.

        Args:
            image (np.ndarray): Input image in HxWxC format.

        Returns:
            list: List of detected ObjectBBox objects.
        """
        image = np.squeeze(image)
        assert isinstance(image, np.ndarray) and image.dtype == np.uint8, "Not a valid image"
        shape = image.shape
        assert len(shape) == 3 and shape[-1] == 3, "Input is not in HxWxC format"
        original_shape = shape[:2]

        # Preprocess
        inp = self._preprocess_input(image)
        # Infer
        out_l = self.ort_sess.run([self.output_name], {self.input_name: inp})
        if len(out_l) != 1:
            print("Error: Infering YOLOv11 failed.")
            return None
        bbox_l = self._postprocess_output(out_l[0], original_shape)
        return bbox_l

def test():
    model = YOLOv11("yolo11s.onnx")
    test_img = Path("test_imgs/pexels-jose-mueses-540180-1280560.jpg")
    img = cv2.imread(test_img)
    assert img is not None, "Test img not found"
    
    bbox_l = model.detect(img)
    for bbox in bbox_l:
        img = bbox.draw(img)
    
    cv2.imwrite(str(test_img.parent / test_img.stem) + "_out.jpg", img) 

if __name__ == "__main__":
    test()