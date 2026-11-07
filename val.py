from argparse import ArgumentParser
from pathlib import Path
from ultralytics import YOLO

def main():
    p=ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--device", default="0")
    p.add_argument("--project", default="runs/val")
    p.add_argument("--data", required=True)
    p.add_argument("--split", choices=("val","test"), default="test")
    a=p.parse_args()
    if not a.weights.is_file():
        raise FileNotFoundError(a.weights)
    model=YOLO(str(a.weights.resolve()), task="detect")
    model.val(data=a.data, split=a.split, imgsz=640, batch=1, device=a.device, project=a.project)

if __name__ == "__main__":
    main()
