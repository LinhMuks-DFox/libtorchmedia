# LibTorchMedia

### Author:
Mux
### Description
A C++ lib, implement torchaudio, torchvision in C++, merge these lib into libtorchmedia

### Dependencies
* Docker

### Usage
For both C++ and Python development, build the Docker container first:

1. Navigate to the appropriate directory:
    ```bash
    cd docker/[cpu/gpu]
    ```
2. Build the Docker container:
    ```bash
    docker compose build
    ```
3. Download dependencies:
    ```bash
    ./download_dep.sh
    ```
    This script will download `fmt` and `libtorch`, and use `poetry` to install Python dependencies.

### VS Code Container Development Support
Configuration files for VS Code container development are placed in the `.devcontainer/` and `.vscode/` directories.


### Adding Python Dependencies
This template uses `poetry` as the package manager for Python. To add a new library, use:
```bash
poetry add <your_lib>
```

### Running Python Code
Use `poetry` to run Python code:
```bash
poetry run python <your_code>
```