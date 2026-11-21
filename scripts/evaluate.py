import _common
import argparse
from pathlib import Path
from dancer.utils import load_config
from dancer.evaluation import evaluate

if __name__ == "__main__":
    p=argparse.ArgumentParser(description="Calculate MECR/PMP/ASW, draw UMAPs and write report.html")
    p.add_argument("--config",required=True)
    a=p.parse_args(); cfg=load_config(a.config)
    _common.logging_to(Path(cfg["paths"]["results"])/"evaluation.log")
    evaluate(cfg)
