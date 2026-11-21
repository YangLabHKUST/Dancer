import _common
import argparse
import json
import time
from pathlib import Path
from dancer.utils import load_config, log, save_json
from dancer.selection import choose_devices, select
from dancer.training import train
from dancer.export import export

if __name__ == "__main__":
    p=argparse.ArgumentParser(description="Select programs, train Dancer and export corrected counts")
    p.add_argument("--config",required=True)
    gpu=p.add_mutually_exclusive_group()
    gpu.add_argument("--gpus",type=int,default=1,help="Number of visible GPUs to use for selection; 0 uses CPU")
    gpu.add_argument("--devices",help="Comma-separated visible GPU indices, e.g. 0,2,3")
    p.add_argument("--stage",choices=["all","select","train","export"],default="all")
    p.add_argument("--resume",action="store_true",help="Resume selection/training from this results directory")
    p.add_argument("--checkpoint",help="Checkpoint path for --stage export")
    a=p.parse_args(); cfg=load_config(a.config)
    root=Path(cfg["paths"]["results"])
    _common.logging_to(root/"run.log")
    devices=choose_devices(a.gpus,a.devices)
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
