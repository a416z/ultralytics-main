from ultralytics import YOLO

model = YOLO(r"D:\yolo\runs\visdrone_yolo11n_640_e100\weights\epoch90.pt")
print(model.task)
print(model.names)
print(sum(p.numel()for p in model.model.parameters()))