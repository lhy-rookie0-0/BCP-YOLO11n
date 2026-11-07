# BCP-YOLO11n

Custom Ultralytics source for urban street-tree detection, combining a PP-LCNet-based backbone, CBAM before SPPF, and four weighted BiFPN fusion nodes.

## Source and license

The bundled Ultralytics version is recorded in ultralytics/__init__.py. Upstream copyright notices and AGPL-3.0 license are retained. See LICENSE and UPSTREAM_README.md.

Model configuration: ultralytics/cfg/models/11/yolo11-PPLCNet+BiFPN+CBAM.yaml. Custom components: ultralytics/nn/Addmodules. Parser: ultralytics/nn/tasks.py. The configuration has four fusion nodes with 2, 2, 3, 2 inputs.

## Environment

The manuscript environment uses Python 3.10, PyTorch 2.7.0, torchvision 0.22.0, CUDA 12.8 and Ubuntu 22.04. In an isolated Python 3.10 environment, run from this repository:

    python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
    python -m pip install -r requirements.txt
    python -m pip install -e . --no-deps

Choose an appropriate PyTorch build for other hardware. Scripts accept --device cpu.

## Dataset

Download https://github.com/lhy-rookie0-0/RT_dataset and reconstruct its archive parts. Preserve the supplied split. Edit configs/rt_dataset.yaml to set path to your absolute dataset root. Expected directories are images/train, images/val, images/test and corresponding labels/train, labels/val, labels/test. The configuration records all eight classes in their expected order. Labels use normalized YOLO class/x_center/y_center/width/height format.

## Training

Supply an official YOLO11n initialization checkpoint. Compatible tensors are transferred; unmatched layers use their implementation initialization.

    python train.py --data /absolute/path/data.yaml --weights /absolute/path/yolo11n.pt --seed 0

Repeat with --seed 1 and --seed 2. Settings are 640 input, batch 32, SGD, maximum 300 epochs, patience 30, initial learning rate 0.01, final factor 0.05, momentum 0.937, weight decay 0.0005, deterministic execution, HSV 0.015/0.7/0.4, translation 0.1, scale 0.5, horizontal flip 0.5, Mosaic 1.0. Output is runs/train/bcp_yolo11n_seed*. Pass --project to change output root. Checkpoint selection follows the bundled trainer.

## Validation and prediction

Use a checkpoint trained with this architecture and class order.

    python val.py --data /absolute/path/data.yaml --weights /absolute/path/best.pt --split test
    python predict.py --weights /absolute/path/best.pt --source /absolute/path/images --conf 0.25

Validation uses 640 input and batch 1. Prediction confidence 0.25 is a visualization default, not a claim about manuscript figure settings.

## Release scope

This release packages the supplied local source configuration, environment declarations and executable entry points. Trained checkpoints are managed separately. Source syntax was checked during packaging. Training, inference and numerical agreement with manuscript results have not been verified in the packaging environment. Metric reproduction requires corresponding experiment checkpoints, fixed data split and original evaluation settings.
