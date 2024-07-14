import numpy as np
import soundfile as sf

# 生成一个1秒的440Hz正弦波
samplerate = 44100
duration = 10.0  # seconds
frequency = 440.0  # Hz
t = np.linspace(0., duration, int(samplerate * duration))
audio = 0.5 * np.sin(2. * np.pi * frequency * t)

# 保存为WAV文件
sf.write('test.wav', audio, samplerate)
print("test.wav created")