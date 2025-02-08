import argparse
import cv2
import numpy as np
from pathlib import Path
from YOLOv11_ONNX_Wrapper import YOLOv11

def process_video(input_video_path, output_video_path, model, save):
    """
    Process a video file and apply YOLOv11 object detection on each frame.

    Args:
        input_video_path (str): Path to the input video file.
        output_video_path (str): Path to save the output video.
        model (YOLOv11): Initialized YOLOv11 model for detection.
        save (bool): Whether to save the output video.
    """
    cap = cv2.VideoCapture(input_video_path)
    assert cap.isOpened(), f"Error: Cannot open video file {input_video_path}"

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            output_video_path, fourcc, fps, (frame_width, frame_height)
        )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        bbox_l = model.detect(frame)
        for bbox in bbox_l:
            frame = bbox.draw(frame)

        if save:
            out.write(frame)

        cv2.imshow(input_video_path.name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    if save:
        out.release()
        print(f"Processed video saved at {output_video_path}")

def main():
    parser = argparse.ArgumentParser(description="YOLOv11 Video Object Detection")
    parser.add_argument("input_video", type=str, help="Path to input video file")
    parser.add_argument("--save", action="store_true", help="Save the processed video")
    args = parser.parse_args()

    input_video_path = Path(args.input_video)
    output_video_path = input_video_path.parent / (input_video_path.stem + "_out.mp4")

    model = YOLOv11("yolo11s.onnx")
    process_video(input_video_path, output_video_path, model, args.save)

if __name__ == "__main__":
    main()
