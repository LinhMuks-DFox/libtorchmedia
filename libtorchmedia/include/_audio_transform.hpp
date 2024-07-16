#pragma once
#ifndef _LIB_TORCH_MEDIA_AUDIO_TRANSFOMR_HPP
#define _LIB_TORCH_MEDIA_AUDIO_TRANSFOMR_HPP
#include <torch/nn/module.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/TensorImpl.h>
#include <cmath>
#include <torch/serialize/input-archive.h>
#include <torch/torch.h>
#include <sox.h>
#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <set>
#include "_audio_functional.hpp"
namespace audio4torch::transform{
    
    class Spectrogram : public torch::nn::Module {
    private:
        audio4torch::functional::spectrogram_option option;
    public:
        auto forward(torch::Tensor signal) -> torch::Tensor {
            return audio4torch::functional::spectrogram(signal, this->option);
        }
    };

}
#endif // _LIB_TORCH_MEDIA_AUDIO_TRANSFOMR_HPP