#!/usr/bin/env python3
"""Compare CUDA-generated PGM outputs against a CPU reference implementation."""

import argparse
from pathlib import Path


def read_pgm(path):
    with path.open("rb") as file:
        magic = file.readline().strip()
        if magic != b"P5":
            raise ValueError(f"{path} is not a binary PGM P5 file")

        tokens = []
        while len(tokens) < 3:
            line = file.readline()
            if not line:
                raise ValueError(f"{path} has an incomplete PGM header")
            if line.startswith(b"#"):
                continue
            tokens.extend(line.split())

        width, height, max_value = map(int, tokens[:3])
        if max_value != 255:
            raise ValueError(f"{path} has unsupported max value {max_value}")

        pixels = file.read()
        expected = width * height
        if len(pixels) != expected:
            raise ValueError(f"{path} has {len(pixels)} pixels; expected {expected}")
        return width, height, pixels


def output_name(input_path, mode):
    return f"{input_path.stem}_{mode}.pgm"


def reference_invert(pixels):
    return bytes(255 - value for value in pixels)


def reference_threshold(pixels, threshold):
    return bytes(255 if value >= threshold else 0 for value in pixels)


def reference_sobel(pixels, width, height):
    output = bytearray(width * height)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            top_left = pixels[(y - 1) * width + (x - 1)]
            top = pixels[(y - 1) * width + x]
            top_right = pixels[(y - 1) * width + (x + 1)]
            left = pixels[y * width + (x - 1)]
            right = pixels[y * width + (x + 1)]
            bottom_left = pixels[(y + 1) * width + (x - 1)]
            bottom = pixels[(y + 1) * width + x]
            bottom_right = pixels[(y + 1) * width + (x + 1)]

            gx = -top_left + top_right - 2 * left + 2 * right - bottom_left + bottom_right
            gy = -top_left - 2 * top - top_right + bottom_left + 2 * bottom + bottom_right
            output[y * width + x] = min(255, abs(gx) + abs(gy))
    return bytes(output)


def reference_pixels(mode, pixels, width, height, threshold):
    if mode == "sobel":
        return reference_sobel(pixels, width, height)
    if mode == "invert":
        return reference_invert(pixels)
    if mode == "threshold":
        return reference_threshold(pixels, threshold)
    raise ValueError(f"unsupported mode: {mode}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True, help="Directory containing source PGM files.")
    parser.add_argument("--output_dir", required=True, help="Directory containing CUDA output PGM files.")
    parser.add_argument("--mode", choices=["sobel", "invert", "threshold"], default="sobel")
    parser.add_argument("--threshold", type=int, default=128)
    parser.add_argument("--max_images", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    input_files = sorted(Path(args.input_dir).glob("*.pgm"))
    if args.max_images > 0:
        input_files = input_files[: args.max_images]
    if not input_files:
        raise SystemExit("No input PGM files found")

    checked = 0
    for input_path in input_files:
        width, height, input_pixels = read_pgm(input_path)
        output_path = Path(args.output_dir) / output_name(input_path, args.mode)
        out_width, out_height, output_pixels = read_pgm(output_path)
        if (width, height) != (out_width, out_height):
            raise SystemExit(f"Dimension mismatch for {output_path}")

        expected_pixels = reference_pixels(
            args.mode, input_pixels, width, height, args.threshold
        )
        if output_pixels != expected_pixels:
            mismatches = sum(
                1 for expected, actual in zip(expected_pixels, output_pixels) if expected != actual
            )
            raise SystemExit(f"{output_path} differs from CPU reference at {mismatches} pixels")
        checked += 1

    print(f"Verified {checked} {args.mode} output image(s) against the CPU reference.")


if __name__ == "__main__":
    main()
