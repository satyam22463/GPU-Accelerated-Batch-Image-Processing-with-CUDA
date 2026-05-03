# Project Presentation Outline

Use this outline for a 5-10 minute capstone demonstration video.

## 1. Goal

Show a CUDA batch image-processing program that runs one selected GPU kernel across many grayscale images in a single command-line execution.

## 2. Why This Project

Image processing is a practical GPU workload because the same operation can be applied independently to many pixels. The project uses a simple PGM format so the focus stays on CUDA memory transfers, kernel launches, timing, and batch execution instead of image-library setup.

## 3. Repository Tour

- `src/main.cu`: command-line parsing, PGM input/output, CUDA memory management, kernels, timing, and logging.
- `scripts/generate_pgm_dataset.py`: creates reproducible input batches.
- `scripts/run_sample.sh`: one-command build and sample run.
- `scripts/verify_outputs.py`: compares CUDA outputs with a CPU reference.
- `proof/`: place committed execution evidence here after running on CUDA hardware.

## 4. CUDA Techniques

- One CUDA thread maps to one output pixel.
- Sobel mode uses a 2D grid and 16x16 thread blocks.
- Invert and threshold modes use a 1D grid over all pixels.
- The program uses `cudaMalloc`, `cudaMemcpy`, kernel launches, `cudaGetLastError`, CUDA events, and `cudaFree`.

## 5. Demonstration Commands

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

## 6. Results To Show

- Terminal output listing the CUDA device, mode, image count, total megapixels, and total GPU kernel time.
- `output/sobel/run_log.txt`, which records per-image GPU kernel timing.
- A few input and output PGM images.
- The verifier output proving the CUDA image files match the CPU reference.

## 7. Challenges And Lessons Learned

- Handling PGM parsing correctly matters before any GPU work can be trusted.
- Boundary pixels in Sobel require explicit handling.
- GPU timing should focus on kernel execution, while the full pipeline also includes file I/O and memory transfers.
- Proof artifacts are part of the engineering work because reviewers need to see that the software handled a batch, not only one image.

## 8. Next Steps

- Process RGB images or PNG/JPEG inputs through a small image library.
- Reuse GPU buffers across images to reduce allocation overhead.
- Add CUDA streams to overlap transfers and kernel execution.
- Compare CPU and GPU end-to-end throughput on larger datasets.
