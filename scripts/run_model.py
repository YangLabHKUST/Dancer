import _common
import argparse
import json
import time
from pathlib import Path
from dancer.utils import load_config, log, save_json
from dancer.selection import select
from dancer.devices import choose_devices
from dancer.training import train
from dancer.export import export

if __name__ == "__main__":
    p=argparse.ArgumentParser(description="Select programs, train Dancer and export corrected counts")
    p.add_argument("--config",required=True)
    gpu=p.add_mutually_exclusive_group()
    gpu.add_argument("--gpus",type=int,help="Number of idle GPUs to use (default: auto-select up to 5)")
    gpu.add_argument("--devices",help="Comma-separated visible GPU indices, e.g. 0,2,3")
    gpu.add_argument("--cpu",action="store_true",help="Run selection, training and export on the CPU")
    p.add_argument("--stage",choices=["all","select","train","export"],default="all")
    p.add_argument("--resume",action="store_true",help="Resume selection/training from this results directory")
    p.add_argument("--checkpoint",help="Checkpoint path for --stage export")
    a=p.parse_args(); cfg=load_config(a.config)
    root=Path(cfg["paths"]["results"])
    _common.logging_to(root/"run.log")
    idle=cfg.get("gpu_selection",{})
    devices=choose_devices(a.gpus,a.devices,cpu=a.cpu,
        max_auto=min(5,len(cfg["program_selection"]["repeat_seeds"])),
        max_used_memory_mib=idle.get("max_used_memory_mib",512),
        max_utilization_percent=idle.get("max_utilization_percent",5))
    log("run_started",stage_requested=a.stage,devices=devices)
    started=time.monotonic()
    if a.stage in ["all","select"]:
        selected=select(cfg,devices,resume=a.resume)
        log("selection_complete",seconds=time.monotonic()-started)
    if a.stage in ["all","train"]:
        path=root/"program_selection/selected_programs.json"
        selected=json.loads(path.read_text())["program_counts"]
        train(cfg,selected,device=devices[0],resume=a.resume)
    if a.stage in ["all","export"]:
        export(cfg,device=devices[0],checkpoint=a.checkpoint)
    save_json(root/(a.stage+"_completed.json"),dict(stage=a.stage,seconds=time.monotonic()-started,devices=devices))
    log("run_complete",seconds=time.monotonic()-started,results=str(root))
