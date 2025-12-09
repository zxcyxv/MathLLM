import subprocess
import sys

print("=" * 50)
print("AIMO3 H100 GPU Test")
print("=" * 50)

# nvidia-smi
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print(result.stdout)

# PyTorch
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {name}")
    print(f"Memory: {mem:.1f} GB")

    if "H100" in name:
        print("\n>>> SUCCESS: H100 via API!")
    else:
        print(f"\n>>> Got: {name}")
