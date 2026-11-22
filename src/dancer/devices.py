"""Select visible, idle NVIDIA GPUs without starting model work."""
import csv
import math
import subprocess


def _query_smi(query):
    try:
        result = subprocess.run(
            ["nvidia-smi", query, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        raise RuntimeError(
            f"Cannot check idle GPUs: nvidia-smi {query} failed: {str(detail).strip()[:500]}. "
            "Use --devices to explicitly "
            "select allocated GPUs, or --cpu."
        ) from exc
    return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))


def _uuid(value):
    return str(value).strip().lower().removeprefix("gpu-")


def choose_devices(number=None, explicit=None, cpu=False, *, max_auto=5,
                   max_used_memory_mib=512, max_utilization_percent=5):
    if sum((number is not None, explicit is not None, bool(cpu))) > 1:
        raise ValueError("Use only one of --gpus, --devices or --cpu")
    if cpu:
        return ["cpu"]
    if number is not None and number < 1:
        raise ValueError("--gpus must be positive; use --cpu for CPU execution")
    if max_auto < 1:
        raise ValueError("Automatic GPU selection requires at least one task")

    import torch

    available = torch.cuda.device_count()
    if explicit is not None:
        try:
            devices = [int(x) for x in explicit.split(",")]
        except ValueError as exc:
            raise ValueError("--devices must be comma-separated visible GPU indices") from exc
        if len(set(devices)) != len(devices) or any(x < 0 or x >= available for x in devices):
            raise ValueError("--devices must contain distinct visible CUDA device indices")
        return [f"cuda:{i}" for i in devices]
    if not available:
        raise RuntimeError("No visible CUDA GPUs. Use --cpu for CPU execution.")
    if not (math.isfinite(max_used_memory_mib) and max_used_memory_mib >= 0
            and math.isfinite(max_utilization_percent) and 0 <= max_utilization_percent <= 100):
        raise ValueError("Invalid gpu_selection idle thresholds")

    # Query before get_device_properties initializes CUDA contexts of this process.
    rows = _query_smi("--query-gpu=uuid,memory.used,memory.free,utilization.gpu")
    processes = _query_smi("--query-compute-apps=gpu_uuid,pid")
    busy = {}
    for row in processes:
        if len(row) != 2 or not row[1].strip().isdigit():
            raise RuntimeError(f"Cannot determine GPU compute-process usage: unexpected row {row!r}; use --devices or --cpu")
        busy.setdefault(_uuid(row[0]), []).append(row[1].strip())
    states = {}
    for row in rows:
        if len(row) != 4:
            raise RuntimeError(f"Unexpected nvidia-smi GPU row {row!r}; use --devices or --cpu")
        uid = _uuid(row[0])
        reasons = []
        if uid in busy:
            reasons.append("compute processes present (PIDs: " + ", ".join(busy[uid]) + ")")
        try:
            used, free, utilization = map(float, row[1:])
        except ValueError:
            reasons.append(f"unavailable metrics: used/free/utilization={row[1:]!r}")
            states[uid] = (0, reasons)
            continue
        if not all(math.isfinite(x) and x >= 0 for x in (used, free, utilization)):
            reasons.append(f"invalid metrics: used={used}, free={free}, utilization={utilization}")
        if used > max_used_memory_mib:
            reasons.append(f"used memory {used:g} MiB exceeds {max_used_memory_mib:g} MiB")
        if utilization > max_utilization_percent:
            reasons.append(f"GPU utilization {utilization:g}% exceeds {max_utilization_percent:g}%")
        states[uid] = (free, reasons)

    # CUDA indices may be reordered or masked; join by UUID, never physical index.
    candidates = []
    diagnostics = []
    for i in range(available):
        try:
            uid = _uuid(getattr(torch.cuda.get_device_properties(i), "uuid", ""))
        except RuntimeError as exc:
            diagnostics.append(f"cuda:{i}: cannot read CUDA device properties: {exc}")
            continue
        if uid not in states:
            diagnostics.append(f"cuda:{i}: UUID {uid or '(missing)'} cannot be matched to nvidia-smi metrics")
            continue
        free, reasons = states[uid]
        if reasons:
            diagnostics.append(f"cuda:{i}: " + "; ".join(reasons))
        else:
            candidates.append((free, i))
            diagnostics.append(f"cuda:{i}: idle ({free:g} MiB free)")
    candidates.sort(key=lambda item: (-item[0], item[1]))
    count = min(5, max_auto, len(candidates)) if number is None else number
    if not candidates or len(candidates) < count:
        requested = "at least one" if number is None else str(number)
        raise RuntimeError(
            f"Requested {requested} idle GPUs; found {len(candidates)} among {available} visible. "
            "Wait for GPUs, use --devices to explicitly select allocated GPUs, or --cpu.\n"
            + "\n".join(diagnostics)
        )
    return [f"cuda:{i}" for _, i in candidates[:count]]
