"""Allow the example commands to run from a checkout or an installed package."""
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))


class Tee:
    def __init__(self, stream, file):
        self.stream, self.file = stream, file

    def write(self, text):
        self.stream.write(text); self.file.write(text); self.file.flush()

    def flush(self):
        self.stream.flush(); self.file.flush()

    def isatty(self):
        return False


def logging_to(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file = path.open("a", buffering=1)
    sys.stdout = Tee(sys.stdout, file)
    sys.stderr = Tee(sys.stderr, file)
