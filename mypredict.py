from pathlib import Path

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

model = YOLO(r"D:\yolo\runs\visdrone_yolo11n_640_e100\weights\epoch90.pt")
model.predict(
    source=r"D:\yolo\datasets\VisDrone\images\val",
    #source=0,
    project=BASE_DIR / "runs" / "detect",
    save=True,
    show=False,
)
