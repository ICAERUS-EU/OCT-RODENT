#!/usr/bin/env bash
set -euo pipefail

mkdir -p detection_dataset/opencv detection_dataset/yolo detection_dataset/yolo-ms

for i in $(seq 0 30); do
	idx=$(printf "%03d" "$i")
	input_raw="results/raw/output_${idx}.raw"
	echo "Processing ${input_raw}"

	python run_pipeline.py \
		--source opencv \
		--input "${input_raw}" \
		--save 1 \
		--save-path "detection_dataset/opencv/output_${idx}.mp4" \
		--blob 1

	python run_pipeline.py \
		--source opencv \
		--input "${input_raw}" \
		--save 1 \
		--save-path "detection_dataset/yolo/output_${idx}.mp4" \
		--yolo 1

	python run_pipeline.py \
		--source opencv \
		--input "${input_raw}" \
		--save 1 \
		--save-path "detection_dataset/yolo-ms/output_${idx}.mp4" \
		--yolo 1 \
		--model "models/yolo11n-finetuned-ms-best.pt"
done
