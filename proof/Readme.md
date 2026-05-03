# Proof Artifacts

This directory is for proof that the CUDA code executed in the lab environment.

Recommended files to commit after running on a CUDA-capable machine:

- `run_log.txt` copied from the output directory.
- 3-5 processed `.pgm` files from the output directory.
- An optional screenshot of the terminal showing the run command and summary.

Example commands:

```bash
python3 scripts/generate_pgm_dataset.py --output data/sample --count 120 --width 192 --height 192
make
./batch_image_processor --input_dir data/sample --output_dir output/sobel --mode sobel
python3 scripts/verify_outputs.py --input_dir data/sample --output_dir output/sobel --mode sobel
cp output/sobel/run_log.txt proof/run_log.txt
cp output/sobel/image_0000_sobel.pgm proof/
cp output/sobel/image_0001_sobel.pgm proof/
cp output/sobel/image_0002_sobel.pgm proof/
```

For full credit, make sure the committed proof shows one execution over many images, not a separate one-image run. The `run_log.txt` file should list the CUDA device, selected mode, image count, per-image GPU timing rows, total pixels, and total GPU kernel time.

The local development machine used to create this repository did not have `nvcc` installed, so CUDA execution proof should be generated in the course lab or another CUDA-capable environment.
