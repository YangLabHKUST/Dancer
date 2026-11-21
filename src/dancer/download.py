"""Download the public breast example and extract only model inputs."""
from pathlib import Path
import shutil
import zipfile
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm.auto import tqdm
from .utils import log, save_json


def fetch(url, destination, probe_bytes=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not probe_bytes:
        log("download_exists", file=str(destination))
        return destination
    partial = destination.with_suffix(destination.suffix + (".probe" if probe_bytes else ".part"))
    offset = 0 if probe_bytes else (partial.stat().st_size if partial.exists() else 0)
    headers = {"Accept-Encoding": "identity", "User-Agent": "Dancer/0.1 (research data downloader)"}
    if probe_bytes:
        headers["Range"] = f"bytes=0-{probe_bytes-1}"
    elif offset:
        headers["Range"] = f"bytes={offset}-"
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=4, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])))
    with session, session.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
        if response.status_code == 416 and offset:
            total = response.headers.get("Content-Range", "").split("/")[-1]
            if total.isdigit() and int(total) == offset:
                partial.replace(destination)
                return destination
        response.raise_for_status()
        if "text/html" in response.headers.get("Content-Type", "").lower():
            raise RuntimeError(f"The server returned an HTML page instead of data: {url}")
        if response.status_code != 206:
            offset = 0
        elif offset and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise RuntimeError("Server returned an unexpected byte range; the existing partial file is unchanged")
        size = int(response.headers.get("Content-Length", 0))
        log("download_started", url=url, status=response.status_code, resume_bytes=offset, content_length=size)
        received = 0
        with partial.open("ab" if offset else "wb") as out, tqdm(total=(offset+size) or None, initial=offset, unit="B", unit_scale=True, desc=destination.name) as bar:
            for chunk in response.iter_content(1024 * 1024):
                if probe_bytes:
                    chunk = chunk[:max(0, probe_bytes-received)]
                out.write(chunk)
                received += len(chunk)
                bar.update(len(chunk))
                if probe_bytes and received >= probe_bytes:
                    break
        if not probe_bytes and size and received != size:
            raise IOError("Incomplete download; rerun the command to resume")
    if probe_bytes:
        log("download_probe_complete", received_bytes=received, file=str(partial))
        return partial
    partial.replace(destination)
    return destination


def extract_inputs(archive, destination):
    """Flatten only the matrix and cell table; do not extract images/transcripts."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    extracted = []
    with zipfile.ZipFile(archive) as bundle:
        for basename in ("cell_feature_matrix.h5", "cells.parquet"):
            matches = [n for n in bundle.namelist() if Path(n).name == basename and not n.startswith("__MACOSX/")]
            if len(matches) != 1:
                raise ValueError(f"Expected one {basename} inside {archive}; found {matches}")
            target = destination / basename
            tmp = target.with_suffix(target.suffix + ".tmp")
            with bundle.open(matches[0]) as src, tmp.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024*1024)
            tmp.replace(target)
            extracted.append(dict(member=matches[0], output=str(target)))
    return extracted


def download(cfg, probe_bytes=None):
    root = Path(cfg["paths"]["raw"])
    sources = cfg["download"]
    jobs = [("xenium_url", root/"downloads/xenium_outs.zip"),
            ("reference_url", root/"reference/filtered_feature_bc_matrix.h5"),
            ("annotation_url", root/"annotations/Cell_Barcode_Type_Matrices.xlsx")]
    manifest = []
    for key, target in jobs:
        result = fetch(sources[key], target, probe_bytes)
        manifest.append(dict(url=sources[key], path=str(result), bytes=result.stat().st_size))
    if not probe_bytes:
        extracted = extract_inputs(jobs[0][1], root/"xenium")
        save_json(root/"download_manifest.json", dict(files=manifest, extracted=extracted))
    else:
        save_json(root/"download_probe.json", manifest)
