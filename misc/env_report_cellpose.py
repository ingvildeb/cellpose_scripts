"""
Print a quick environment report for OS, Python, PyTorch, and CUDA/GPU availability.

Usage:
- Run `python misc/env_report_cellpose.py` before troubleshooting install/runtime issues.
"""

import sys
import platform
import torch
import cellpose

print("===== SYSTEM =====")
print("OS:", platform.platform())
print("Python:", sys.version)

print("\n===== PYTORCH =====")
print("Torch version:", torch.__version__)
print("Torch CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:", torch.cuda.get_device_name(i))
        print("  Capability:", torch.cuda.get_device_capability(i))
