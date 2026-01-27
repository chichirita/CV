import cv2
from pathlib import Path

IMG_DIR = "data/MOT17-02/img1"
OUT = "data/MOT17-02/mot17.mp4"
FPS = 25

def main():
    imgs = sorted(Path(IMG_DIR).glob("*.jpg"))
    if not imgs:
        print("No images found!")
        return

    sample = cv2.imread(str(imgs[0]))
    h, w, _ = sample.shape

    writer = cv2.VideoWriter(
        OUT,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (w, h)
    )

    for img_path in imgs:
        frame = cv2.imread(str(img_path))
        writer.write(frame)

    writer.release()
    print("Video saved at:", OUT)

if __name__ == "__main__":
    main()
