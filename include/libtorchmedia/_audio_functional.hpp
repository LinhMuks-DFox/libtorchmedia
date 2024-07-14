#pragma once
#ifndef _LIB_TORCH_MEDIA_AUDIO_FUNCTIONAL_HPP
#define _LIB_TORCH_MEDIA_AUDIO_FUNCTIONAL_HPP
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

namespace audio4torch::functional {
    
    enum convolve_mode {
        real,
        full
    };

    auto inline _pad_sequence(torch::Tensor tensor, int length, const torch::nn::functional::PadFuncOptions& options) -> torch::Tensor {
        return torch::nn::functional::pad(tensor, options);
    
    }
    auto convolve(torch::Tensor signal, torch::Tensor filter, convolve_mode mode) -> torch::Tensor {
        // Broadcast signal and filter to match shapes except the last dimension
        if (!signal.sizes().equals(filter.sizes())) {
            std::vector<int64_t> new_shape;
            for (size_t i = 0; i < signal.sizes().size() - 1; ++i) {
                new_shape.push_back(std::max(signal.size(i), filter.size(i)));
            }
            signal = signal.broadcast_to(torch::IntArrayRef(new_shape).vec()).contiguous();
            filter = filter.broadcast_to(torch::IntArrayRef(new_shape).vec()).contiguous();
        }

        int64_t signal_length = signal.size(-1);
        int64_t filter_length = filter.size(-1);

        // Pad signal and filter based on mode
        torch::Tensor padded_signal;
        torch::Tensor padded_filter;

        if (mode == full) {
            int64_t pad_length = filter_length - 1;
            auto pad_options = torch::nn::functional::PadFuncOptions({pad_length, pad_length}).mode(torch::kConstant).value(0);
            padded_signal = _pad_sequence(signal, signal_length + 2 * pad_length, pad_options);
        } else {
            padded_signal = signal;
        }

        padded_filter = filter.unsqueeze(0).unsqueeze(0); // Add batch and channel dimensions

        // Apply 1D convolution
        padded_signal = padded_signal.unsqueeze(0).unsqueeze(0); // Add batch and channel dimensions

        torch::Tensor convolved = torch::nn::functional::conv1d(padded_signal, padded_filter);

        convolved = convolved.squeeze(0).squeeze(0); // Remove batch and channel dimensions

        // Apply convolve mode adjustments
        auto apply_convolve_mode = [&]() -> torch::Tensor {
            if (mode == full) {
                return convolved;
            } else {
                int64_t trim_length = filter_length - 1;
                return convolved.slice(/*dim=*/-1, /*start=*/trim_length, /*end=*/signal_length + trim_length);
            }
        };

        return apply_convolve_mode();
    }


    auto amplitude_to_DB(torch::Tensor signal, float amin, float db_multiplier, float topdb, bool apply_topdb) -> torch::Tensor{
            // Ensure amin is greater than 0 to avoid log of zero
        amin = std::max(amin, std::numeric_limits<float>::min());

        // Convert amplitude to power
        torch::Tensor power = torch::pow(signal, 2.0);

        // Apply logarithm
        torch::Tensor db = 10.0 * torch::log10(torch::clamp(power, amin, std::numeric_limits<float>::max()));

        // Apply multiplier
        db = db * db_multiplier;

        // Apply top_db limit if needed
        if (apply_topdb) {
            float max_db = db.max().item<float>();
            db = torch::max(db, torch::tensor(max_db - topdb));
        }
        return db;
    }

    struct spectrogram_option {
        int _pad = 0;
        torch::Tensor _window = {};
        int _n_fft = 400;
        int _hop_length = 200;
        int _win_length = 400;
        float _power = 2.0;
        bool _normalized = false;
        std::string _normalize_method = "window"; // window, frame_length
        bool _center = true;
        std::string _pad_mode = "reflect";
        bool _onesided = true;
        bool _return_complex = false; // when true, power becomes optional;

        auto pad(int p) -> spectrogram_option& {
            _pad = p;
            return *this;
        }

        auto window(torch::Tensor w) -> spectrogram_option& {
            _window = w;
            return *this;
        }

        auto n_fft(int n) -> spectrogram_option& {
            _n_fft = n;
            return *this;
        }

        auto hop_length(int h) -> spectrogram_option& {
            _hop_length = h;
            return *this;
        }

        auto win_length(int w) -> spectrogram_option& {
            _win_length = w;
            return *this;
        }

        auto power(float p) -> spectrogram_option& {
            _power = p;
            return *this;
        }

        auto normalized(bool n) -> spectrogram_option& {
            _normalized = n;
            return *this;
        }

        auto normalize_method(const std::string& n) -> spectrogram_option& {
            _normalize_method = n;
            return *this;
        }

        auto center(bool c) -> spectrogram_option& {
            _center = c;
            return *this;
        }

        auto pad_mode(const std::string& p) -> spectrogram_option& {
            _pad_mode = p;
            return *this;
        }

        auto onesided(bool o) -> spectrogram_option& {
            _onesided = o;
            return *this;
        }

        auto return_complex(bool r) -> spectrogram_option& {
            _return_complex = r;
            return *this;
        }
    };

    auto spectrogram(torch::Tensor signal, spectrogram_option option) -> torch::Tensor {
        int pad_amount = option._pad;
        if (pad_amount > 0) {
            signal = torch::constant_pad_nd(signal, {pad_amount, pad_amount}, 0);
        }

        int n_fft = option._n_fft;
        int hop_length = option._hop_length;
        int win_length = option._win_length;
        torch::Tensor window = option._window.defined() ? option._window : torch::hann_window(win_length, torch::TensorOptions().dtype(signal.dtype()));

        auto spec_f = torch::stft(
            signal,
            n_fft,
            hop_length,
            win_length,
            window,
            option._center,
            option._pad_mode,
            option._onesided,
            option._return_complex
        );

        if (option._return_complex) {
            return spec_f;
        } else {
            auto spec = torch::pow(torch::abs(spec_f), option._power);
            if (option._normalized) {
                if (option._normalize_method == "window") {
                    spec /= window.pow(2).sum().sqrt();
                } else if (option._normalize_method == "frame_length") {
                    spec /= win_length;
                }
            }
            return spec;
        }
    }
}
#endif // _LIB_TORCH_MEDIA_AUDIO_FUNCTIONAL_HPP