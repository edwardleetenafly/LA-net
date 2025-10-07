import torch
from src.utils.determinism import seed_everything
from src.utils.device import get_device

# Set seed & deterministic modes once
seed_everything(42, deterministic=True, cudnn_benchmark=False)

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("Device name:", torch.cuda.get_device_name(0))
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    print("MPS (Apple Silicon) available: True")
else:
    print("Using CPU")

# Tiny op to confirm everything is wired
import torch.nn.functional as F
x = torch.randn(1, 1, 8, 8, device=get_device())
k = torch.ones(1, 1, 3, 3, device=get_device()) / 9.0
y = F.conv2d(x, k, padding=1)
print("Conv OK, output mean:", y.mean().item())
