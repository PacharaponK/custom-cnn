"""Shared file/experiment utilities; no model or dataset logic hidden here."""
import argparse
import csv
import hashlib
import json
import random
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TZ = timezone(timedelta(hours=7))

def now():
    return datetime.now(TZ).isoformat()

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))

def configuration(path="config.json"):
    cfg_path = ROOT / path
    cfg = read_json(cfg_path)
    if not cfg["run_id"]:
        cfg["run_id"] = datetime.now(TZ).strftime("f10-%Y%m%d-%H%M%S") + "-s6610110190-w64h36-b4-e30"
        write_json(cfg_path, cfg)
    return cfg

def output(cfg):
    path = ROOT / "result" / cfg["run_id"]
    path.mkdir(parents=True, exist_ok=True)
    for name in ["figures", "logs"]:
        (path / name).mkdir(exist_ok=True)
    return path

def record(cfg, step, state="completed", **details):
    path = output(cfg) / "pipeline-history.json"
    history = read_json(path) if path.exists() else []
    history.append({"time": now(), "step": step, "state": state,
                    "argv": sys.argv, "run_id": cfg["run_id"], **details})
    write_json(path, history)

def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_model(cfg, device):
    import torch
    from model import CompactUNet
    checkpoint = ROOT / "checkpoints" / cfg["run_id"] / "last.pt"
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    if saved["manifest_hash"] != sha256(ROOT / "data" / "manifest.csv"):
        raise ValueError("Checkpoint dataset manifest does not match")
    location_keys = {"source_dataset", "vault_assignment"}
    saved_experiment = {k: v for k, v in saved["config"].items() if k not in location_keys}
    current_experiment = {k: v for k, v in cfg.items() if k not in location_keys}
    if saved_experiment != current_experiment:
        raise ValueError("Checkpoint configuration does not match")
    model = CompactUNet().to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, checkpoint
