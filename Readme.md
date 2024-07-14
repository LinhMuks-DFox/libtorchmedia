# Deep Learning Dev Template

### Author:
Mux
### Description
A development template for deep learning projects using C++ and Python.

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

### Development Rules
1. Place all C++ source files in directories starting with `pp`, such as `pp_src`, except for the `include` directory.
2. Place all Python source files in directories starting with `py`, such as `py_src`.
3. Build C++ executables in the `build` directory. **Do not** build them in the workspace root directory.

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