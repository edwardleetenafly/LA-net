"""Minimal training stub to validate wiring."""
from pathlib import Path
import yaml

from src.utils.determinism import seed_everything
from src.utils.device import get_device

def main(cfg_path: str = "configs/default.yaml"):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    seed_everything(
        cfg.get("seed", 42),
        cfg.get("deterministic", True),
        cfg.get("cudnn_benchmark", False),
    )
    device = get_device()
    print("Config loaded:", cfg_path)
    print("Device:", device)
    print("Core settings:", {k: cfg[k] for k in ["epochs", "batch_size", "learning_rate"]})

if __name__ == "__main__":
    main()
