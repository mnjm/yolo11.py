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
