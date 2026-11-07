"""Train BCP-YOLO11n on the RT dataset.

Run on AutoDL from this directory after activating the rt-yolo environment:
    python train_bcp_yolo11n_rt.py --seed 0
"""

from argparse import ArgumentParser
from pathlib import Path

from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_YAML = PROJECT_DIR / "ultralytics/cfg/models/11/yolo11-PPLCNet+BiFPN+CBAM.yaml"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/train")
    return parser.parse_args()


def main():
    args = parse_args()
    for path in (args.data, MODEL_YAML, args.weights):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    model = YOLO(str(MODEL_YAML), task="detect")
    model.load(str(args.weights.resolve()))
    model.train(
        data=str(args.data.resolve()),
        epochs=300,
        patience=30,
        imgsz=640,
        batch=32,
        workers=8,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.05,
        momentum=0.937,
        weight_decay=0.0005,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        seed=args.seed,
        deterministic=True,
        device=args.device,
        project=args.project,
        name=f"bcp_yolo11n_seed{args.seed}",
    )


if __name__ == "__main__":
    main()
