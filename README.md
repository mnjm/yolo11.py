# YOLOv11 ONNX Inference Wrapper

Python wrapper for running YOLOv11 object detection using an ONNX model with ONNXRuntime, Including optimized post processing with class targeted NMS.

## Usage

```sh
pip install -r requirements.txt
```
If you have GPU, uncomment `onnxruntime` in `requirements.txt` and uncomment `onnxruntime-gpu` line

## Running on Image

```python
from yolo_wrapper import YOLOv11
import cv2
from pathlib import Path

model = YOLOv11("yolo11s.onnx")
test_img = Path("test_imgs/sample.jpg")
img = cv2.imread(str(test_img))

bbox_list = model.detect(img)
for bbox in bbox_list:
    img = bbox.draw(img)

cv2.imwrite(str(test_img.parent / test_img.stem) + "_out.jpg", img)
```

## Running on video

You can take a look at `test_vid.py`.

To run it, use
```sh
python3 test_vid.py <input_video.mp4> <--save>
```
`--save` will save the output video.

## Class Targeted NMS

You can pass a function or callable to filter valid classes, making NMS slightly efficient. Example

```python
valid_class_d = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
}

model = YOLOv11(
    model_path="yolo11s.onnx",
    valid_class_checker=lambda lbl_id, _: 1 <= lbl_id <= 8 # detect only vehicles
    # (or)
    # valid_class_checker=lambda lbl_id, lbl: lbl_id in valid_class_d and lbl == valid_class_d[lbl_id]
)
```
To get all {class_id, name} pairs
```python
print("\n".join([ f"{k}: {v}" for k,v in model.get_class_id_name_pairs().items() ]))
```