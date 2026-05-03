# CUDA Batch Image Processing

This project processes a large batch of grayscale images with CUDA kernels. It was built for the CUDA at Scale independent project rubric: the code uses GPU computation, accepts command-line arguments, includes build/run support files, and produces execution logs and processed image artifacts that can be committed as proof.

## What It Does

The program reads a directory of binary PGM (`P5`) images, copies each image to the GPU, runs one CUDA image-processing kernel, and writes the processed images to an output directory.

Supported modes:

- `sobel`: edge detection with a 3x3 Sobel operator.
- `invert`: invert grayscale intensities.
- `threshold`: produce a binary image using `--threshold`.

The repository includes `scripts/generate_pgm_dataset.py`, which creates hundreds of synthetic image inputs without external downloads. The generated dataset is intentionally simple and reproducible so reviewers can run the project in a lab machine without extra libraries.

## Repository Layout

```text
CudaBatchImageProcessing/
  CMakeLists.txt
  Makefile
  README.md
  scripts/
    generate_pgm_dataset.py
    run_sample.sh
    verify_outputs.py
  src/
    main.cu
  proof/
    README.md
  PRESENTATION_OUTLINE.md
  PROJECT_DESCRIPTION.md
```

## Build

### CMake

```bash
cmake -S . -B build
cmake --build build --config Release
```

Run the executable from:

```bash
./build/batch_image_processor
```

### Makefile

```bash
make
```

Run the executable from:

```bash
./batch_image_processor
```

## Generate Sample Data

Create 120 small grayscale images:

```bash
python3 scripts/generate_pgm_dataset.py --output data/sample --count 120 --width 192 --height 192
```

Create 30 larger images:

```bash
python3 scripts/generate_pgm_dataset.py --output data/large --count 30 --width 2048 --height 2048
```

## Run

Sobel edge detection over the generated batch:

```bash
./batch_image_processor --input_dir data/sample --output_dir output/sobel --mode sobel
```

Threshold mode:

```bash
./batch_image_processor --input_dir data/sample --output_dir output/threshold --mode threshold --threshold 120
```

Limit a run to the first 20 images:

```bash
./batch_image_processor --input_dir data/sample --output_dir output/sobel20 --mode sobel --max_images 20
```

Skip writing images and only measure GPU processing:

```bash
./batch_image_processor --input_dir data/large --output_dir output/large --mode sobel --write_outputs false
```

## Command-Line Arguments

| Argument | Required | Description |
| --- | --- | --- |
| `--input_dir <path>` | Yes | Directory containing `.pgm` images. |
| `--output_dir <path>` | Yes | Directory where processed `.pgm` images and `run_log.txt` are written. |
| `--mode <sobel|invert|threshold>` | No | GPU operation to run. Default: `sobel`. |
| `--threshold <0-255>` | No | Threshold value for threshold mode. Default: `128`. |
| `--max_images <n>` | No | Process only the first `n` images after sorting by filename. |
| `--write_outputs <true|false>` | No | Enable or disable processed image output. Default: `true`. |

## One-Command Sample

On Linux lab machines with CUDA installed:

```bash
chmod +x scripts/run_sample.sh
./scripts/run_sample.sh
```

This builds the program, generates 120 inputs, runs Sobel filtering, and writes `output/sobel/run_log.txt`.

## Verify Correctness

After a CUDA run writes output images, compare them against a CPU reference implementation:

```bash
python3 scripts/verify_outputs.py --input_dir data/sample --output_dir output/sobel --mode sobel
```

The verifier supports the same modes as the CUDA program. For threshold mode, pass the same threshold used in the CUDA run:

```bash
python3 scripts/verify_outputs.py --input_dir data/sample --output_dir output/threshold --mode threshold --threshold 120
```

## GPU Computation

The CUDA work is in `src/main.cu`.

- `SobelKernel` computes edge magnitude from neighboring pixels.
- `InvertKernel` computes `255 - pixel`.
- `ThresholdKernel` maps pixels to either `0` or `255`.

Each input image is transferred to device memory with `cudaMemcpy`, processed by a CUDA kernel, and copied back to host memory. CUDA events record GPU processing time.

## Proof of Execution

After running the sample command, commit proof artifacts such as:

- `output/sobel/run_log.txt`
- a few processed images from `output/sobel`
- a terminal screenshot showing the command and summary output
- the output from `scripts/verify_outputs.py`

The `proof/README.md` file lists exactly what to capture for the course submission.

The repository also includes `PRESENTATION_OUTLINE.md` for the required 5-10 minute demonstration.

## Short Project Description

This project demonstrates batch image processing on the GPU. It uses CUDA kernels to process many grayscale images in a single program run and supports three image-processing operations. I used a simple PGM pipeline so the project focuses on GPU memory movement, kernel launches, timing, and batch execution rather than image-library setup. The Sobel mode is the most significant path because it uses neighboring pixels and boundary handling instead of a one-pixel transformation.

A slightly longer submission-ready description is available in `PROJECT_DESCRIPTION.md`.
