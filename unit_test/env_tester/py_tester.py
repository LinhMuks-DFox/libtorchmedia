try:
    import torch
    import torchvision
    import torchaudio
    print("Torch, torchvision, and torchaudio installed successfully.")
    print(f"CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("Torch, torchvision, or torchaudio not installed")

try:
    import numpy as np
    print("NumPy installed successfully.")
    x = np.array([1, 2, 3])
    print(f"NumPy array: {x}")
except ImportError:
    print("NumPy not installed")

try:
    import matplotlib.pyplot as plt
    print("Matplotlib installed successfully.")
    plt.plot([1, 2, 3], [1, 4, 9])
    plt.title("Test Plot")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
except ImportError:
    print("Matplotlib not installed")

try:
    import tqdm
    print("TQDM installed successfully.")
    for i in tqdm.tqdm(range(10)):
        pass
except ImportError:
    print("TQDM not installed")

try:
    import seaborn as sns
    print("Seaborn installed successfully.")
    sns.set_theme()
    data = sns.load_dataset("penguins")
    sns.pairplot(data, hue="species")
    plt.show()
except ImportError:
    print("Seaborn not installed")

try:
    from sklearn import datasets
    print("Scikit-learn installed successfully.")
    iris = datasets.load_iris()
    print(f"Iris dataset loaded: {iris.data.shape}")
except ImportError:
    print("Scikit-learn not installed")

try:
    import soundfile as sf
except ImportError:
    print("SoundFile not installed")