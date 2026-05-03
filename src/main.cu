#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kBlockSize = 16;

enum class Mode {
  kSobel,
  kInvert,
  kThreshold,
};

struct Options {
  fs::path input_dir;
  fs::path output_dir;
  Mode mode = Mode::kSobel;
  int threshold = 128;
  int max_images = 0;
  bool write_outputs = true;
};

struct Image {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> pixels;
};

std::string ModeName(Mode mode) {
  switch (mode) {
    case Mode::kSobel:
      return "sobel";
    case Mode::kInvert:
      return "invert";
    case Mode::kThreshold:
      return "threshold";
  }
  return "unknown";
}

void CheckCuda(cudaError_t result, const char* expression, const char* file,
               int line) {
  if (result != cudaSuccess) {
    std::ostringstream message;
    message << "CUDA error at " << file << ":" << line << " for " << expression
            << ": " << cudaGetErrorString(result);
    throw std::runtime_error(message.str());
  }
}

#define CHECK_CUDA(expression) CheckCuda((expression), #expression, __FILE__, __LINE__)

bool ParseBool(const std::string& value) {
  if (value == "true" || value == "1" || value == "yes") {
    return true;
  }
  if (value == "false" || value == "0" || value == "no") {
    return false;
  }
  throw std::runtime_error("Invalid boolean value: " + value);
}

Mode ParseMode(const std::string& value) {
  if (value == "sobel") {
    return Mode::kSobel;
  }
  if (value == "invert") {
    return Mode::kInvert;
  }
  if (value == "threshold") {
    return Mode::kThreshold;
  }
  throw std::runtime_error("Invalid mode: " + value);
}

void PrintUsage(const char* program) {
  std::cerr
      << "Usage: " << program
      << " --input_dir <path> --output_dir <path> [options]\n\n"
      << "Options:\n"
      << "  --mode <sobel|invert|threshold>  GPU operation. Default: sobel\n"
      << "  --threshold <0-255>              Threshold mode cutoff. Default: 128\n"
      << "  --max_images <n>                 Process only first n sorted images\n"
      << "  --write_outputs <true|false>     Write processed PGM files. Default: true\n";
}

Options ParseArgs(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto require_value = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + name);
      }
      return argv[++i];
    };

    if (arg == "--input_dir") {
      options.input_dir = require_value(arg);
    } else if (arg == "--output_dir") {
      options.output_dir = require_value(arg);
    } else if (arg == "--mode") {
      options.mode = ParseMode(require_value(arg));
    } else if (arg == "--threshold") {
      options.threshold = std::stoi(require_value(arg));
    } else if (arg == "--max_images") {
      options.max_images = std::stoi(require_value(arg));
    } else if (arg == "--write_outputs") {
      options.write_outputs = ParseBool(require_value(arg));
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }

  if (options.input_dir.empty() || options.output_dir.empty()) {
    throw std::runtime_error("--input_dir and --output_dir are required");
  }
  if (options.threshold < 0 || options.threshold > 255) {
    throw std::runtime_error("--threshold must be between 0 and 255");
  }
  if (options.max_images < 0) {
    throw std::runtime_error("--max_images must be non-negative");
  }
  return options;
}

std::string ReadToken(std::istream& input) {
  std::string token;
  while (input >> token) {
    if (!token.empty() && token[0] == '#') {
      std::string ignored;
      std::getline(input, ignored);
      continue;
    }
    return token;
  }
  throw std::runtime_error("Unexpected end of PGM header");
}

Image ReadPgm(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Unable to open input image: " + path.string());
  }

  if (ReadToken(input) != "P5") {
    throw std::runtime_error("Only binary PGM P5 files are supported: " +
                             path.string());
  }

  Image image;
  image.width = std::stoi(ReadToken(input));
  image.height = std::stoi(ReadToken(input));
  const int max_value = std::stoi(ReadToken(input));
  if (image.width <= 0 || image.height <= 0 || max_value != 255) {
    throw std::runtime_error("Invalid PGM dimensions or max value: " +
                             path.string());
  }

  input.get();
  image.pixels.resize(static_cast<std::size_t>(image.width) * image.height);
  input.read(reinterpret_cast<char*>(image.pixels.data()), image.pixels.size());
  if (input.gcount() != static_cast<std::streamsize>(image.pixels.size())) {
    throw std::runtime_error("PGM pixel data is truncated: " + path.string());
  }
  return image;
}

void WritePgm(const fs::path& path, const Image& image) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("Unable to open output image: " + path.string());
  }
  output << "P5\n" << image.width << " " << image.height << "\n255\n";
  output.write(reinterpret_cast<const char*>(image.pixels.data()),
               image.pixels.size());
}

std::vector<fs::path> ListPgmFiles(const fs::path& input_dir, int max_images) {
  std::vector<fs::path> files;
  for (const fs::directory_entry& entry : fs::directory_iterator(input_dir)) {
    if (entry.is_regular_file() && entry.path().extension() == ".pgm") {
      files.push_back(entry.path());
    }
  }

  std::sort(files.begin(), files.end());
  if (max_images > 0 && static_cast<int>(files.size()) > max_images) {
    files.resize(max_images);
  }
  return files;
}

__global__ void InvertKernel(const std::uint8_t* input, std::uint8_t* output,
                             int pixel_count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < pixel_count) {
    output[index] = 255 - input[index];
  }
}

__global__ void ThresholdKernel(const std::uint8_t* input, std::uint8_t* output,
                                int pixel_count, int threshold) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < pixel_count) {
    output[index] = input[index] >= threshold ? 255 : 0;
  }
}

__global__ void SobelKernel(const std::uint8_t* input, std::uint8_t* output,
                            int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }

  if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
    output[y * width + x] = 0;
    return;
  }

  const int top_left = input[(y - 1) * width + (x - 1)];
  const int top = input[(y - 1) * width + x];
  const int top_right = input[(y - 1) * width + (x + 1)];
  const int left = input[y * width + (x - 1)];
  const int right = input[y * width + (x + 1)];
  const int bottom_left = input[(y + 1) * width + (x - 1)];
  const int bottom = input[(y + 1) * width + x];
  const int bottom_right = input[(y + 1) * width + (x + 1)];

  const int gx = -top_left + top_right - 2 * left + 2 * right - bottom_left +
                 bottom_right;
  const int gy = -top_left - 2 * top - top_right + bottom_left + 2 * bottom +
                 bottom_right;
  const int magnitude = min(255, abs(gx) + abs(gy));
  output[y * width + x] = static_cast<std::uint8_t>(magnitude);
}

float ProcessImageOnGpu(const Image& input_image, Image* output_image,
                        Mode mode, int threshold) {
  output_image->width = input_image.width;
  output_image->height = input_image.height;
  output_image->pixels.resize(input_image.pixels.size());

  std::uint8_t* device_input = nullptr;
  std::uint8_t* device_output = nullptr;
  cudaEvent_t start;
  cudaEvent_t stop;

  const std::size_t byte_count = input_image.pixels.size();
  CHECK_CUDA(cudaMalloc(&device_input, byte_count));
  CHECK_CUDA(cudaMalloc(&device_output, byte_count));
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaMemcpy(device_input, input_image.pixels.data(), byte_count,
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaEventRecord(start));
  if (mode == Mode::kSobel) {
    const dim3 block(kBlockSize, kBlockSize);
    const dim3 grid((input_image.width + block.x - 1) / block.x,
                    (input_image.height + block.y - 1) / block.y);
    SobelKernel<<<grid, block>>>(device_input, device_output,
                                 input_image.width, input_image.height);
  } else {
    constexpr int kThreadsPerBlock = 256;
    const int pixel_count = static_cast<int>(input_image.pixels.size());
    const int blocks = (pixel_count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    if (mode == Mode::kInvert) {
      InvertKernel<<<blocks, kThreadsPerBlock>>>(device_input, device_output,
                                                 pixel_count);
    } else {
      ThresholdKernel<<<blocks, kThreadsPerBlock>>>(
          device_input, device_output, pixel_count, threshold);
    }
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CHECK_CUDA(cudaMemcpy(output_image->pixels.data(), device_output, byte_count,
                        cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaFree(device_input));
  CHECK_CUDA(cudaFree(device_output));
  return elapsed_ms;
}

fs::path OutputPathFor(const fs::path& input_path, const fs::path& output_dir,
                       Mode mode) {
  fs::path filename = input_path.stem();
  filename += "_" + ModeName(mode) + ".pgm";
  return output_dir / filename;
}

std::string DeviceSummary() {
  int device_count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    throw std::runtime_error("No CUDA devices found");
  }

  int device = 0;
  CHECK_CUDA(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  CHECK_CUDA(cudaGetDeviceProperties(&properties, device));

  std::ostringstream summary;
  summary << properties.name << " (compute capability " << properties.major
          << "." << properties.minor << ", "
          << properties.multiProcessorCount << " SMs)";
  return summary.str();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseArgs(argc, argv);
    if (!fs::exists(options.input_dir)) {
      throw std::runtime_error("Input directory does not exist: " +
                               options.input_dir.string());
    }
    fs::create_directories(options.output_dir);

    const std::vector<fs::path> files =
        ListPgmFiles(options.input_dir, options.max_images);
    if (files.empty()) {
      throw std::runtime_error("No .pgm files found in " +
                               options.input_dir.string());
    }

    const std::string device_summary = DeviceSummary();
    std::ofstream log(options.output_dir / "run_log.txt");
    if (!log) {
      throw std::runtime_error("Unable to write run_log.txt");
    }

    std::cout << "CUDA device: " << device_summary << "\n";
    std::cout << "Mode: " << ModeName(options.mode) << "\n";
    std::cout << "Images: " << files.size() << "\n";

    log << "CUDA Batch Image Processing Run\n";
    log << "Device: " << device_summary << "\n";
    log << "Mode: " << ModeName(options.mode) << "\n";
    log << "Input directory: " << fs::absolute(options.input_dir).string()
        << "\n";
    log << "Output directory: " << fs::absolute(options.output_dir).string()
        << "\n";
    log << "Image count: " << files.size() << "\n";
    log << "Filename,Width,Height,Pixels,GPU kernel ms\n";

    double total_gpu_ms = 0.0;
    std::uint64_t total_pixels = 0;
    for (const fs::path& file : files) {
      const Image input_image = ReadPgm(file);
      Image output_image;
      const float gpu_ms = ProcessImageOnGpu(input_image, &output_image,
                                             options.mode, options.threshold);

      total_gpu_ms += gpu_ms;
      total_pixels += input_image.pixels.size();

      if (options.write_outputs) {
        WritePgm(OutputPathFor(file, options.output_dir, options.mode),
                 output_image);
      }

      log << file.filename().string() << "," << input_image.width << ","
          << input_image.height << "," << input_image.pixels.size() << ","
          << std::fixed << std::setprecision(4) << gpu_ms << "\n";
    }

    const double megapixels = static_cast<double>(total_pixels) / 1.0e6;
    std::cout << "Processed " << files.size() << " images (" << std::fixed
              << std::setprecision(2) << megapixels << " MP) using "
              << ModeName(options.mode) << "\n";
    std::cout << "Total GPU kernel time: " << std::fixed
              << std::setprecision(3) << total_gpu_ms << " ms\n";
    std::cout << "Log: " << (options.output_dir / "run_log.txt").string()
              << "\n";

    log << "Total pixels: " << total_pixels << "\n";
    log << "Total megapixels: " << std::fixed << std::setprecision(2)
        << megapixels << "\n";
    log << "Total GPU kernel ms: " << std::fixed << std::setprecision(3)
        << total_gpu_ms << "\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    PrintUsage(argv[0]);
    return EXIT_FAILURE;
  }
}
