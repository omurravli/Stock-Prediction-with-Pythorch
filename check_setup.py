import platform, sys

import matplotlib, sklearn, torch

import numpy as np, pandas as pd


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    
    return torch.device("cpu")

def main() -> None:

    device = get_device()

    print("=" * 50)
    print("System Information")
    print("=" * 50)
    print(f"Operating System: {platform.platform()}")
    print(f"Python Version: {sys.version.split()[0]}")

    print("\n" + "=" * 50)
    print("Package Versions")
    print("=" * 50)
    print(f"Pytorch: {torch.__version__}")
    print(f"Pandas: {pd.__version__}")
    print(f"Numpy: {np.__version__}")
    print(f"Scikit-learn: {sklearn.__version__}")
    print(f"Matplotlib: {matplotlib.__version__}")

    print("\n" + "=" * 50)
    print("Device Information")
    print("=" * 50)
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built:  {torch.backends.mps.is_built()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Selected Device: {device}")

    print("\n" + "=" * 50)
    print("Tensor Test")
    print("=" * 50)


    x = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype = torch.float32,
        device = device
    )

    y = torch.tensor(
        [[2.0, 0.0], [1.0, 2.0]],
        dtype = torch.float32,
        device = device
    )

    result = torch.matmul(x, y)

    print("Input tensor: ")
    print(x)

    print("\n Matrix multiplication result: ")
    print(result)

    print("\n Setup Completed Successfully.")

if __name__ == "__main__":
    main()
