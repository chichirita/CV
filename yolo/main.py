import cv2
import torch
from ultralytics import YOLO
import supervision as sv

VID = "data/MOT17-02/mot17.mp4"
OUT = "out.mp4"

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    yolo = YOLO("yolov8n.pt").to(dev)
    if dev == "cuda":
        yolo.fuse()
        yolo.model.half()

    trk = sv.ByteTrack()
    box_draw = sv.BoxAnnotator(thickness=2)
    lbl_draw = sv.LabelAnnotator()

    cap = cv2.VideoCapture(VID)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        res = yolo(frame, conf=0.4, iou=0.5, verbose=False)[0]
        det = sv.Detections.from_ultralytics(res)
        det = trk.update_with_detections(det)

        labels = [f"id {i}" for i in det.tracker_id]

        frame = box_draw.annotate(frame, det)
        frame = lbl_draw.annotate(frame, det, labels)
        out.write(frame)

    cap.release()
    out.release()
    print("saved:", OUT)

if __name__ == "__main__":
    main()
