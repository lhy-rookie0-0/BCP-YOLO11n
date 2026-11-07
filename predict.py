from argparse import ArgumentParser
from pathlib import Path
from ultralytics import YOLO

def main():
    p=ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--device", default="0")
    p.add_argument("--project", default="runs/predict")
    p.add_argument("--source", required=True)
    p.add_argument("--conf", type=float, default=0.25)
    a=p.parse_args()
    if not a.weights.is_file():
        raise FileNotFoundError(a.weights)
    model=YOLO(str(a.weights.resolve()), task="detect")
    model.predict(source=a.source, conf=a.conf, imgsz=640, device=a.device, save=True, project=a.project)

if __name__ == "__main__":
    main()
