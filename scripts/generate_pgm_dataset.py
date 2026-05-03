#!/usr/bin/env python3
"""Generate a reproducible batch of binary PGM images for CUDA processing."""

import argparse
import math
import random
from pathlib import Path


def pixel_value(x, y, index, width, height):
    """Create a textured grayscale value with edges and smooth gradients."""
    x_norm = x / max(1, width - 1)
    y_norm = y / max(1, height - 1)
    wave = 80.0 * math.sin((x_norm * 8.0 + index * 0.13) * math.pi)
    wave += 60.0 * math.cos((y_norm * 6.0 - index * 0.07) * math.pi)
    diagonal = 90.0 if (x + index * 3) % max(8, width // 8) < width // 20 else 0.0
    circle_x = width * (0.35 + 0.25 * math.sin(index * 0.11))
    circle_y = height * (0.45 + 0.20 * math.cos(index * 0.09))
    radius = min(width, height) * (0.12 + 0.03 * (index % 5))
    circle = 70.0 if (x - circle_x) ** 2 + (y - circle_y) ** 2 < radius ** 2 else 0.0
    value = 95.0 + wave + diagonal + circle + 45.0 * x_norm + 30.0 * y_norm
    return max(0, min(255, int(value)))


def write_pgm(path, width, height, index):
    random.seed(index)
    data = bytearray(width * height)
    for y in range(height):
        for x in range(width):
            value = pixel_value(x, y, index, width, height)
            value += random.randint(-8, 8)
            data[y * width + x] = max(0, min(255, value))

    with path.open("wb") as file:
        file.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        file.write(data)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Directory to write PGM files.")
    parser.add_argument("--count", type=int, default=120, help="Number of images.")
    parser.add_argument("--width", type=int, default=192, help="Image width.")
    parser.add_argument("--height", type=int, default=192, help="Image height.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.count <= 0 or args.width <= 0 or args.height <= 0:
        raise SystemExit("count, width, and height must be positive")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for index in range(args.count):
        path = output_dir / f"image_{index:04d}.pgm"
        write_pgm(path, args.width, args.height, index)

    print(
        f"Generated {args.count} PGM images in {output_dir} "
        f"({args.width}x{args.height})"
    )


if __name__ == "__main__":
    main()
