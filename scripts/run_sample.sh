#!/usr/bin/env bash
set -euo pipefail

python3 scripts/generate_pgm_dataset.py --output data/sample --count 120 --width 192 --height 192

if command -v cmake >/dev/null 2>&1; then
  cmake -S . -B build
  cmake --build build --config Release
  ./build/batch_image_processor --input_dir data/sample --output_dir output/sobel --mode sobel
else
  make
  ./batch_image_processor --input_dir data/sample --output_dir output/sobel --mode sobel
fi
