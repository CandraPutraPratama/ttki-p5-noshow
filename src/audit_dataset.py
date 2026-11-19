"""Audit dataset sebelum eksperimen (buku petunjuk bagian 5.1).

    python src/audit_dataset.py --data data/KaggleV2-May-2016.csv

Mencetak ringkasan dan menulis results/audit.json. Jalankan ini lebih dulu dan
catat hasilnya di bagian Methods (jumlah baris dibuang, rentang tanggal, dll).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import clean, load_raw  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    raw = load_raw(args.data)
    audit = {
        "n_rows_raw": int(len(raw)),
        "n_columns": int(raw.shape[1]),
        "missing_per_column": {k: int(v) for k, v in raw.isna().sum().items()},
        "age_min": float(raw["age"].min()),
        "age_max": float(raw["age"].max()),
        "handicap_values": sorted(int(v) for v in raw["handicap"].dropna().unique()),
        "n_neighbourhoods": int(raw["neighbourhood"].nunique()),
        "target_values": sorted(str(v) for v in raw["no_show"].unique()),
    }
    _, report = clean(raw)
    audit["cleaning"] = report

    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
