#pragma once
#include <sstream>
#include <string>
#ifndef LIB_TORCH_EXT_UTILS_HPP
#define LIB_TORCH_EXT_UTILS_HPP
namespace torch_ext {
template <class TorchData>
auto inline to_string(const TorchData& obj) -> std::string
{
    std::stringstream ss;
    ss << obj << std::flush;
    return ss.str();
}
}
#endif