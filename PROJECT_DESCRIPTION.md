# Short Project Description

This project performs batch image processing with CUDA. It reads many grayscale PGM images from an input directory, sends each image to the GPU, applies a selected CUDA kernel, and writes the processed result plus a run log.

The main algorithm is Sobel edge detection. Each CUDA thread computes one output pixel by reading the 3x3 neighborhood around that pixel, applying horizontal and vertical Sobel filters, and writing the clamped edge magnitude. The project also includes simpler invert and threshold kernels to demonstrate command-line selection of GPU operations.

I chose generated PGM images so the reviewer can create hundreds of inputs without needing third-party image libraries or downloading a dataset. This keeps the project focused on CUDA concepts: batch processing, host-to-device and device-to-host memory transfer, kernel launch geometry, boundary handling, and GPU timing with CUDA events.

The program is run from the command line and supports options for input directory, output directory, processing mode, threshold value, maximum image count, and whether output images should be written. Build and sample execution are supported through both CMake and a Makefile.

For correctness checking, I included a Python CPU reference verifier. After the CUDA program writes processed images, the verifier recomputes the same operation on the CPU and reports whether the output files match. This gives reviewers a simple way to confirm that the GPU path produced the expected data.

The most important lesson from the project is that GPU work is only one part of an end-to-end image-processing pipeline. Reading files, allocating memory, transferring data, launching kernels, timing execution, and preserving reproducible proof artifacts all matter when turning a CUDA kernel into a usable batch-processing application.
