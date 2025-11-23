# Rodent thermal detection toolkit

Fine-tuning notebooks, pretrained weights, and a modular pipeline for running Ultralytics YOLOv11n and/or hotspot detection on thermal imagery collected from FLIR cameras.

## Layout

```
.
├── data/                  # (untracked) local dataset structure
│   ├── images/{train, valid, test}
│   ├── labels/{train, valid, test}
│   └── data.yaml         
├── environments/          # Conda environments (gpu-yolo.yml, spin-yolo.yml)
├── framework/             # Detection pipeline core, sources, and plugins
│   ├── pipeline.py
│   ├── sources/
│   │   ├── opencv_source.py
│   │   ├── raw_file_source.py
│   │   └── spinnaker_source.py
│   └── plugins/
│       ├── temperature_scale_plugin.py
│       ├── yolo_plugin.py
│       └── opencv_blob_plugin.py
├── models/                # YOLO checkpoints 
├── other/                 # Utility scripts
│   ├── frame_playback.py
│   ├── raw_to_jpg.py
│   ├── save_raw_frames.py
│   └── save_detections.sh
├── results/               # Saved MP4 outputs and raw 14-bit thermal frames
│   runs/                  # YOLO finetuning results 
│       ├── yolo11n-finetuned
│       └── yolo11n-finetuned-ms
├── finetune_yolo.ipynb    # YOLOv11n fine-tuning notebook
├── finetune.yaml          # Training configuration
└── run_pipeline.py        # CLI entry point for live/replay inference
```

## Components

- `finetune_yolo.ipynb` contains the YOLOv11n finetuning script using the [Roboflow dataset](https://universe.roboflow.com/panav2/rodent-thermal/dataset/2). Download and export locally into `data/` following the folder structure in the layout section. The notebook documents the full training run for the multi-scale model.

- `environments/gpu-yolo.yml` contains GPU training dependencies.

- `environments/spin-yolo.yml` contains dependencies for Spinnaker SDK bindings for IR camera control, streaming, and real-time YOLO inference. Requires Spinnaker SDK to be installed separately.

- `models/yolo11n-finetuned-best.pt` Finetuned YOLOv11n checkpoint.

- `models/yolo11n-finetuned-ms-best.pt` Finetuned YOLOv11n checkpoint trained with multi-scale augmentation.

- The pipeline in `framework/pipeline.py` wires together input frame sources (`sources/`) and plugins (`plugins/`):
  - `yolo_plugin.py` runs Ultralytics YOLOv11 (single-scale or multi-scale) and draws detections.
  - `opencv_blob_plugin.py` maintains rolling median backgrounds, builds motion and hotspot masks with OpenCV (Gaussian blur, morphology, contours), and keeps per-object tracks to label moving vs static objects while suppressing duplicates against YOLO.
  - `temperature_scale_plugin.py` colorizes thermal frames and optionally overlays a temperature bar.
  - `sources/` include `opencv_source.py` for OpenCV capture, `spinnaker_source.py` for FLIR Spinnaker streams, and `raw_file_source.py` for raw thermal playback.

## Running the pipeline

- File or camera via OpenCV: `python run_pipeline.py --source opencv --input data/images/test --yolo 1`
- FLIR AX5 through Spinnaker: `python run_pipeline.py --source spinnaker --yolo 1 --blob 1`

Press ESC or `q` to close the display window. Use `--display 0` for headless runs and `--save 1` to record annotated MP4s under `results/mp4/`. Raw 14-bit dumps land in `results/raw/` when `--raw-save 1` is set. Other options are available via `python run_pipeline.py --help`.

## Datasets

The `rodent_dataset` (available on Zenodo - DOI pending) was created by extracting frames from raw thermal video streams captured using: `python run_pipeline.py --source spinnaker --raw-save 1`, and then running `other/save_raw_frames.py` and `other/raw_to_jpg.py`.

The `detection_dataset` (available on Zenodo - DOI pending) was created by running inference on saved raw streams by running `other/save_detections.sh`.