#include "_audio_functional.hpp"
#include <ATen/ops/arange.h>
#include <audio.hpp>
#include <c10/core/TensorOptions.h>
#include <fmt/core.h>
#include <torch/torch.h>
#include <torch/types.h>
#include <utils.hpp>
int main()
{

    auto x = torch::arange(0, 100, torch::TensorOptions().dtype(torch::kFloat32));
    auto y = torch::tensor({ 1, 0, 1, 0, 1 }, torch::TensorOptions().dtype(torch::kFloat32));
    auto convolved = audio4torch::functional::convolve(x, y, audio4torch::functional::convolve_mode::full);
    fmt::print("{}", torch_ext::to_string(convolved));
    return 0;
}