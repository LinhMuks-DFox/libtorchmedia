#include <torch/torch.h>
#include <sox.h>
#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <set>

namespace audio4torch {
    struct AudioData {
        torch::Tensor data;
        int sample_rate;
    };

    auto load_audio(const std::string &path) -> AudioData {
        // Initialize SoX library
        if (sox_init() != SOX_SUCCESS) {
            throw std::runtime_error("Failed to initialize SoX");
        }

        sox_format_t* format = sox_open_read(path.c_str(), nullptr, nullptr, nullptr);
        if (!format) {
            sox_quit();
            throw std::runtime_error("Failed to open audio file");
        }

        int sample_rate = format->signal.rate;
        size_t num_samples = format->signal.length;
        std::vector<sox_sample_t> samples(num_samples);
        size_t samples_read = sox_read(format, samples.data(), num_samples);
        if (samples_read != num_samples) {
            sox_close(format);
            sox_quit();
            throw std::runtime_error("Failed to read audio samples");
        }

        sox_close(format);
        sox_quit();

        // Convert samples to float
        std::vector<float> float_samples(samples.size());
        std::memcpy(float_samples.data(), samples.data(), samples.size() * sizeof(sox_sample_t));

        for (auto &sample : float_samples) {
            sample /= SOX_SAMPLE_MAX;
        }

        // Create a torch tensor from the samples
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor tensor = torch::from_blob(float_samples.data(), {static_cast<int64_t>(float_samples.size())}, options).clone();

        return {tensor, sample_rate};
    }
}
