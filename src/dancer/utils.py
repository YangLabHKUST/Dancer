"""Configuration, logging, reproducibility and checkpoint utilities."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json
import random
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml


def load_config(path):
    path = Path(path).resolve()
    with path.open() as stream:
        cfg = yaml.safe_load(stream)
    root = (path.parent / cfg.get("project_root", "..")).resolve()
    for key, value in cfg["paths"].items():
        cfg["paths"][key] = str((root / value).resolve())
    cfg["project_root"] = str(root)
    return cfg


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    tmp.replace(path)


def log(stage, **values):
    print(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(), stage=stage, **values), default=str), flush=True)


def seed_all(seed):
    import torch
    import pyro
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@contextmanager
def random_stream(seed, device):
    import torch
    device = torch.device(device)
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        yield


def atomic_torch_save(value, path):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def cpu_store():
    import pyro
    state = pyro.get_param_store().get_state()
    return dict(params={k: v.detach().cpu().clone() for k, v in state["params"].items()}, constraints=state["constraints"].copy())


def restore_store(model, state, device):
    import pyro
    from pyro.params.param_store import param_with_module_name
    params = {n: p.detach().to(device).requires_grad_(True) for n, p in state["params"].items()}
    for n, p in model.spatial_module.named_parameters():
        key = param_with_module_name("dancer_spatial_module", n)
        if key in params:
            params[key] = p
    pyro.get_param_store().set_state(dict(params=params, constraints=state["constraints"]))


def require_empty_stage(folder, resume=False):
    """Prevent accidental overwrites of another run, without restricting inputs."""
    folder = Path(folder)
    if folder.exists() and any(folder.iterdir()) and not resume:
        raise FileExistsError(f"{folder} is not empty. Use --resume, or select another results path in the config.")
    folder.mkdir(parents=True, exist_ok=True)
