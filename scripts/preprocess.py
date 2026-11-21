import _common
import argparse
from dancer.utils import load_config
from dancer.data import preprocess

if __name__ == "__main__":
    p=argparse.ArgumentParser(description="Prepare labeled reference and spatial count matrices")
    p.add_argument("--config",required=True)
    a=p.parse_args()
    preprocess(load_config(a.config))
