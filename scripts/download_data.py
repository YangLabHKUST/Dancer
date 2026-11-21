import _common
import argparse
from dancer.utils import load_config
from dancer.download import download

if __name__ == "__main__":
    p=argparse.ArgumentParser(description="Download the breast cancer example and extract model inputs")
    p.add_argument("--config",required=True)
    p.add_argument("--probe-bytes",type=int,default=None,help="Only test the first N bytes of each download; do not extract")
    a=p.parse_args()
    if a.probe_bytes is not None and a.probe_bytes<=0:
        p.error("--probe-bytes must be positive")
    download(load_config(a.config),a.probe_bytes)
