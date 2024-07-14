#include <iostream>
#include <torch/torch.h>
#include <fmt/core.h>

auto inline to_string(const torch::Tensor & t) {
    std::stringstream ss;
    ss << t;
    return ss.str();
}


int main() {
    torch::Tensor tensor = torch::eye(3);
    fmt::print("Tensor:\n{}\n", to_string(tensor));
    std::cout << "If you see a 3x3 identity matrix above, libtorch is working correctly!" << std::endl;
    return 0;
}