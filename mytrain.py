from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"yolo11n.pt")
    model.train(
    data=r"D:\yolo\datasets\VisDrone\VisDrone.yaml",
    epochs=5,
    imgsz=640,
    batch=16,
    cache=False ,
    workers=0,
)
