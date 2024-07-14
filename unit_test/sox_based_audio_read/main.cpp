#include "../../include/audio.hpp"
int main() {
    try {
        auto loaded = load_audio("/workspace/unit_test/ffmpeg_based_audio_read/test.wav");
        auto audio_tensor = loaded.data;
        std::cout << "Audio tensor loaded successfully: " << audio_tensor.sizes() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}