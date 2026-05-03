CXXFLAGS := -std=c++17 -O3
NVCCFLAGS := -std=c++17 -O3
TARGET := batch_image_processor
SRC := src/main.cu

.PHONY: all clean sample

all: $(TARGET)

$(TARGET): $(SRC)
	nvcc $(NVCCFLAGS) -o $@ $<

sample: $(TARGET)
	python3 scripts/generate_pgm_dataset.py --output data/sample --count 120 --width 192 --height 192
	./$(TARGET) --input_dir data/sample --output_dir output/sobel --mode sobel

clean:
	rm -f $(TARGET)
	rm -rf build output data/sample
