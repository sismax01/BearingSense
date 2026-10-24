# =============================================================================
# BearingSense — CWRU MATLAB (.mat) & CSV Vibration Data Loaders
# Engineered by : SISMAX (sismax01)
# =============================================================================
"""Data loaders for Case Western Reserve University (CWRU) `.mat` files and CSV vibration logs.

Supports:
  1. Official CWRU MATLAB (`.mat`) files containing variable keys like `X097_DE_time`,
     `X105_DE_time`, `X118_DE_time`, `X130_DE_time`, etc., as well as `X097RPM`.
  2. Standard CSV vibration files with an acceleration column (`acceleration`, `de_time`, or `signal`).
  3. Built-in metadata catalog mapping standard CWRU file numbers (e.g., `97.mat`, `105.mat`,
     `118.mat`, `130.mat`, `169.mat`, `185.mat`, `197.mat`, `209.mat`, `222.mat`, `234.mat`)
     to their documented fault class, fault diameter (inches), motor load (HP), and shaft speed (RPM).
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import io as spio

from bearingsense.simulation import FAULT_CLASSES

# Official CWRU 12k Drive End Bearing Fault Data & Normal Baseline metadata catalog
# Source: https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data
CWRU_FILE_CATALOG: Dict[int, Dict[str, object]] = {
    # Normal Baseline Data
    97: {"label": "Normal", "fault_size_in": 0.0, "load_hp": 0, "rpm": 1797.0, "fs": 12000.0},
    98: {"label": "Normal", "fault_size_in": 0.0, "load_hp": 1, "rpm": 1772.0, "fs": 12000.0},
    99: {"label": "Normal", "fault_size_in": 0.0, "load_hp": 2, "rpm": 1750.0, "fs": 12000.0},
    100: {"label": "Normal", "fault_size_in": 0.0, "load_hp": 3, "rpm": 1730.0, "fs": 12000.0},
    # 0.007" Inner Race Fault (IR007)
    105: {"label": "Inner_Race_Fault", "fault_size_in": 0.007, "load_hp": 0, "rpm": 1797.0, "fs": 12000.0},
    106: {"label": "Inner_Race_Fault", "fault_size_in": 0.007, "load_hp": 1, "rpm": 1772.0, "fs": 12000.0},
    107: {"label": "Inner_Race_Fault", "fault_size_in": 0.007, "load_hp": 2, "rpm": 1750.0, "fs": 12000.0},
    108: {"label": "Inner_Race_Fault", "fault_size_in": 0.007, "load_hp": 3, "rpm": 1730.0, "fs": 12000.0},
    # 0.007" Ball Fault (B007)
    118: {"label": "Ball_Fault", "fault_size_in": 0.007, "load_hp": 0, "rpm": 1797.0, "fs": 12000.0},
    119: {"label": "Ball_Fault", "fault_size_in": 0.007, "load_hp": 1, "rpm": 1772.0, "fs": 12000.0},
    120: {"label": "Ball_Fault", "fault_size_in": 0.007, "load_hp": 2, "rpm": 1750.0, "fs": 12000.0},
    121: {"label": "Ball_Fault", "fault_size_in": 0.007, "load_hp": 3, "rpm": 1730.0, "fs": 12000.0},
    # 0.007" Outer Race Fault Centered @ 6:00 (OR007@6)
    130: {"label": "Outer_Race_Fault", "fault_size_in": 0.007, "load_hp": 0, "rpm": 1797.0, "fs": 12000.0},
    131: {"label": "Outer_Race_Fault", "fault_size_in": 0.007, "load_hp": 1, "rpm": 1772.0, "fs": 12000.0},
    132: {"label": "Outer_Race_Fault", "fault_size_in": 0.007, "load_hp": 2, "rpm": 1750.0, "fs": 12000.0},
    133: {"label": "Outer_Race_Fault", "fault_size_in": 0.007, "load_hp": 3, "rpm": 1730.0, "fs": 12000.0},
    # 0.014" Inner Race Fault (IR014)
    169: {"label": "Inner_Race_Fault", "fault_size_in": 0.014, "load_hp": 0, "rpm": 1797.0, "fs": 12000.0},
    170: {"label": "Inner_Race_Fault", "fault_size_in": 0.014, "load_hp": 1, "rpm": 1772.0, "fs": 12000.0},
    171: {"label": "Inner_Race_Fault", "fault_size_in": 0.014, "load_hp": 2, "rpm": 1750.0, "fs": 12000.0},
    172: {"label": "Inner_Race_Fault", "fault_size_in": 0.014, "load_hp": 3, "rpm": 1730.0, "fs": 12000.0},
    # 0.014" Ball Fault (B014)
    185: {"label": "Ball_Fault", "fault_size_in": 0.014, "load_hp": 0, "rpm": 1797.0, "fs": 12000.0},
    186: {"label": "Ball_Fault", "fault_size_in": 0.014, "load_hp": 1, "rpm": 1772.0, "fs": 12000.0},
    187: {"label": "Ball_Fault", "fault_size_in": 0.014, "load_hp": 2, "rpm": 1750.0, "fs": 12000.0},
    188: {"label": "Ball_Fault", "fault_size_in": 0.014, "load_hp": 3, "rpm": 1730.0, "fs": 12000.0},
    # 0.014" Outer Race Fault @ 6:00 (OR014@6)
    197: {"label": "Outer_Race_Fault", "fault_size_in": 0.014, "load_hp": 0, "rpm": 1797.0, "fs": 12000.0},
    198: {"label": "Outer_Race_Fault", "fault_size_in": 0.014, "load_hp": 1, "rpm": 1772.0, "fs": 12000.0},
    199: {"label": "Outer_Race_Fault", "fault_size_in": 0.014, "load_hp": 2, "rpm": 1750.0, "fs": 12000.0},
    200: {"label": "Outer_Race_Fault", "fault_size_in": 0.014, "load_hp": 3, "rpm": 1730.0, "fs": 12000.0},
    # 0.021" Inner Race Fault (IR021)
    209: {"label": "Inner_Race_Fault", "fault_size_in": 0.021, "load_hp": 0, "rpm": 1797.0, "fs": 12000.0},
    210: {"label": "Inner_Race_Fault", "fault_size_in": 0.021, "load_hp": 1, "rpm": 1772.0, "fs": 12000.0},
    211: {"label": "Inner_Race_Fault", "fault_size_in": 0.021, "load_hp": 2, "rpm": 1750.0, "fs": 12000.0},
    212: {"label": "Inner_Race_Fault", "fault_size_in": 0.021, "load_hp": 3, "rpm": 1730.0, "fs": 12000.0},
    # 0.021" Ball Fault (B021)
    222: {"label": "Ball_Fault", "fault_size_in": 0.021, "load_hp": 0, "rpm": 1797.0, "fs": 12000.0},
    223: {"label": "Ball_Fault", "fault_size_in": 0.021, "load_hp": 1, "rpm": 1772.0, "fs": 12000.0},
    224: {"label": "Ball_Fault", "fault_size_in": 0.021, "load_hp": 2, "rpm": 1750.0, "fs": 12000.0},
    225: {"label": "Ball_Fault", "fault_size_in": 0.021, "load_hp": 3, "rpm": 1730.0, "fs": 12000.0},
    # 0.021" Outer Race Fault @ 6:00 (OR021@6)
    234: {"label": "Outer_Race_Fault", "fault_size_in": 0.021, "load_hp": 0, "rpm": 1797.0, "fs": 12000.0},
    235: {"label": "Outer_Race_Fault", "fault_size_in": 0.021, "load_hp": 1, "rpm": 1772.0, "fs": 12000.0},
    236: {"label": "Outer_Race_Fault", "fault_size_in": 0.021, "load_hp": 2, "rpm": 1750.0, "fs": 12000.0},
    237: {"label": "Outer_Race_Fault", "fault_size_in": 0.021, "load_hp": 3, "rpm": 1730.0, "fs": 12000.0},
}


def infer_label_from_filename(filename: str) -> Optional[str]:
    """Infer fault class label from a filename or CWRU file number."""
    stem = Path(filename).stem
    for cls_name in FAULT_CLASSES:
        if cls_name.lower() in stem.lower():
            return cls_name

    match = re.search(r"(?:^|X|_)(\d{2,3})(?:$|_)", stem)
    if match:
        file_num = int(match.group(1))
        if file_num in CWRU_FILE_CATALOG:
            return str(CWRU_FILE_CATALOG[file_num]["label"])

    lower = stem.lower()
    if "normal" in lower or "norm" in lower:
        return "Normal"
    if "outer" in lower or "or0" in lower or "bpfo" in lower:
        return "Outer_Race_Fault"
    if "inner" in lower or "ir0" in lower or "bpfi" in lower:
        return "Inner_Race_Fault"
    if "ball" in lower or re.search(r"(?:^|_)b0\d\d", lower) or "bsf" in lower:
        return "Ball_Fault"

    return None


def load_cwru_mat_file(
    file_path: str | Path,
    channel_preference: str = "DE_time",
    default_fs: float = 12000.0,
    default_rpm: float = 1797.0,
    override_label: Optional[str] = None,
) -> Dict:
    """Load a CWRU-format MATLAB (`.mat`) file and extract its vibration signal and metadata."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"MATLAB file not found: {path}")

    try:
        mat_data = spio.loadmat(str(path))
    except Exception as exc:
        raise ValueError(f"Failed to read MATLAB .mat file '{path}': {exc}") from exc

    user_keys = [k for k in mat_data.keys() if not k.startswith("__")]
    signal_key = None
    for k in user_keys:
        if channel_preference.lower() in k.lower():
            signal_key = k
            break
    if signal_key is None:
        for k in user_keys:
            val = mat_data[k]
            if isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.number) and val.size >= 64:
                signal_key = k
                break

    if signal_key is None:
        raise ValueError(
            f"Could not locate a valid vibration signal array in '{path.name}'. Found keys: {user_keys}"
        )

    raw_sig = np.asarray(mat_data[signal_key], dtype=np.float64).squeeze()
    if raw_sig.ndim != 1 or raw_sig.size < 64:
        raise ValueError(f"Extracted signal '{signal_key}' in '{path.name}' is invalid (shape={raw_sig.shape}).")

    rpm = default_rpm
    for k in user_keys:
        if "rpm" in k.lower():
            rpm_val = np.asarray(mat_data[k], dtype=np.float64).squeeze()
            if rpm_val.size == 1 and float(rpm_val) > 0:
                rpm = float(rpm_val)
                break

    catalog_meta: Dict[str, object] = {}
    match = re.search(r"(\d{2,3})", path.stem)
    if match:
        fnum = int(match.group(1))
        if fnum in CWRU_FILE_CATALOG:
            catalog_meta = CWRU_FILE_CATALOG[fnum]
            if rpm == default_rpm:
                rpm = float(catalog_meta["rpm"])

    label = override_label
    if label is None and "label" in mat_data:
        embedded_lbl = mat_data["label"]
        if isinstance(embedded_lbl, np.ndarray) and embedded_lbl.size > 0:
            label = str(embedded_lbl.squeeze())
    if label is None:
        label = infer_label_from_filename(path.name)
    if label is None:
        label = "Unknown"

    fs = default_fs
    if "fs" in mat_data:
        fs_val = np.asarray(mat_data["fs"], dtype=np.float64).squeeze()
        if fs_val.size == 1 and float(fs_val) > 0:
            fs = float(fs_val)
    elif "fs" in catalog_meta:
        fs = float(catalog_meta["fs"])

    source_type = "cwru_real"
    if "source_type" in mat_data:
        st_val = mat_data["source_type"]
        if isinstance(st_val, np.ndarray) and st_val.size > 0:
            source_type = str(st_val.squeeze())
    elif path.stem.startswith("SYN_"):
        source_type = "synthetic_simulation"

    return {
        "signal": raw_sig,
        "fs": fs,
        "rpm": rpm,
        "label": label,
        "run_id": path.stem,
        "load_hp": int(catalog_meta.get("load_hp", 0)),
        "fault_size_in": float(catalog_meta.get("fault_size_in", 0.0)),
        "source_type": source_type,
        "file_path": str(path),
    }


def load_csv_vibration_file(
    file_path: str | Path,
    fs: float = 12000.0,
    rpm: float = 1797.0,
    override_label: Optional[str] = None,
) -> Dict:
    """Load a 1D vibration signal from a CSV file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV file '{path}' is empty.")

    candidate_cols = ["signal", "acceleration", "de_time", "vibration", "x"]
    selected_col = None
    for c in df.columns:
        if c.strip().lower() in candidate_cols:
            selected_col = c
            break
    if selected_col is None:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_time = [c for c in num_cols if "time" not in c.lower()]
        selected_col = non_time[0] if non_time else (num_cols[0] if num_cols else None)

    if selected_col is None:
        raise ValueError(f"No numeric vibration column found in CSV '{path}'.")

    sig = df[selected_col].dropna().to_numpy(dtype=np.float64)
    if sig.size < 64:
        raise ValueError(f"CSV vibration column '{selected_col}' has fewer than 64 samples ({sig.size}).")

    label = override_label or infer_label_from_filename(path.name) or "Unknown"
    return {
        "signal": sig,
        "fs": fs,
        "rpm": rpm,
        "label": label,
        "run_id": path.stem,
        "load_hp": 0,
        "fault_size_in": 0.0,
        "source_type": "csv_recording",
        "file_path": str(path),
    }


def load_directory_dataset(
    data_dir: str | Path,
    default_fs: float = 12000.0,
    default_rpm: float = 1797.0,
) -> List[Dict]:
    """Load all `.mat` and `.csv` bearing vibration recordings from a directory."""
    dir_path = Path(data_dir)
    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dir_path}")

    records: List[Dict] = []
    for mat_file in sorted(dir_path.glob("*.mat")):
        rec = load_cwru_mat_file(mat_file, default_fs=default_fs, default_rpm=default_rpm)
        if rec["label"] in FAULT_CLASSES:
            records.append(rec)

    for csv_file in sorted(dir_path.glob("*.csv")):
        rec = load_csv_vibration_file(csv_file, fs=default_fs, rpm=default_rpm)
        if rec["label"] in FAULT_CLASSES:
            records.append(rec)

    if not records:
        raise ValueError(
            f"No valid labeled bearing `.mat` or `.csv` recordings found in '{dir_path}'. "
            f"Filenames should match CWRU IDs (e.g., 97.mat, 105.mat, 118.mat, 130.mat) "
            f"or include class names ({FAULT_CLASSES})."
        )
    return records
