"""Pemuatan, pembersihan, dan pembagian data Medical Appointment No Shows (Kaggle).

Semua keputusan pembersihan dicatat dalam `report` supaya bisa dilaporkan di
bagian Methods paper. Tidak ada informasi dari validation/test yang dipakai di
sini: transformasi yang dipelajari dari data (scaler, encoder) dibuat di
`build_preprocessor` dan hanya boleh di-fit pada data train (lewat Pipeline).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Nama kolom di Kaggle punya salah eja (Hipertension, Handcap) dan beberapa
# salinan memakai variasi lain; semuanya dipetakan ke satu nama baku.
_ALIASES = {
    "patientid": "patient_id",
    "appointmentid": "appointment_id",
    "gender": "gender",
    "scheduledday": "scheduled_day",
    "appointmentday": "appointment_day",
    "age": "age",
    "neighbourhood": "neighbourhood",
    "neighborhood": "neighbourhood",
    "scholarship": "scholarship",
    "hipertension": "hypertension",
    "hypertension": "hypertension",
    "diabetes": "diabetes",
    "alcoholism": "alcoholism",
    "handcap": "handicap",
    "handicap": "handicap",
    "smsreceived": "sms_received",
    "noshow": "no_show",
}

REQUIRED = [
    "patient_id", "appointment_id", "gender", "scheduled_day", "appointment_day",
    "age", "neighbourhood", "scholarship", "hypertension", "diabetes",
    "alcoholism", "handicap", "sms_received", "no_show",
]

NUMERIC = ["age", "lead_days", "scheduled_hour", "appt_weekday", "sched_weekday", "handicap"]
BINARY = ["is_female", "scholarship", "hypertension", "diabetes", "alcoholism", "sms_received"]
CATEGORICAL = ["neighbourhood"]


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {}
    for col in df.columns:
        key = re.sub(r"[^a-z]", "", str(col).lower())
        if key in _ALIASES:
            rename[col] = _ALIASES[key]
    df = df.rename(columns=rename)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"Kolom wajib tidak ditemukan: {missing}. Kolom yang ada: {list(df.columns)}. "
            "Periksa apakah ini salinan Kaggle yang benar (joniarroba/noshowappointments)."
        )
    return df[REQUIRED].copy()


def _to_binary_target(s: pd.Series) -> pd.Series:
    mapping = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}
    out = s.astype(str).str.strip().str.lower().map(mapping)
    if out.isna().any():
        bad = s[out.isna()].unique()[:5]
        raise ValueError(f"Nilai target tidak dikenali: {bad}")
    return out.astype(int)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Bersihkan data dan buat fitur turunan dari kolom penjadwalan.

    lead_days = tanggal AppointmentDay dikurangi tanggal ScheduledDay (keduanya dipotong ke
    tengah malam, jadi jam tidak berpengaruh). sms_received bersifat observasional dan bisa
    dikirim setelah penjadwalan; dampaknya diuji terpisah lewat --exclude-sms.
    """
    report: dict = {"rows_raw": int(len(df))}
    d = df.copy()

    d["scheduled_day"] = pd.to_datetime(d["scheduled_day"], utc=True).dt.tz_localize(None)
    d["appointment_day"] = pd.to_datetime(d["appointment_day"], utc=True).dt.tz_localize(None)
    d["target"] = _to_binary_target(d["no_show"])

    dup = d.duplicated(subset="appointment_id", keep="first")
    report["dropped_duplicate_appointment_id"] = int(dup.sum())
    d = d[~dup]

    d["lead_days"] = (d["appointment_day"].dt.normalize() - d["scheduled_day"].dt.normalize()).dt.days
    bad_lead = d["lead_days"] < 0
    report["dropped_negative_lead_days"] = int(bad_lead.sum())
    d = d[~bad_lead]

    bad_age = (d["age"] < 0) | (d["age"] > 115)
    report["dropped_invalid_age"] = int(bad_age.sum())
    d = d[~bad_age]

    d["is_female"] = (d["gender"].astype(str).str.upper().str.strip() == "F").astype(int)
    d["scheduled_hour"] = d["scheduled_day"].dt.hour
    d["appt_weekday"] = d["appointment_day"].dt.weekday
    d["sched_weekday"] = d["scheduled_day"].dt.weekday
    d["neighbourhood"] = d["neighbourhood"].astype(str)

    report["rows_clean"] = int(len(d))
    report["no_show_rate_clean"] = float(d["target"].mean())
    report["unique_patients"] = int(d["patient_id"].nunique())
    report["rows_with_repeated_patient"] = int(d.duplicated("patient_id", keep=False).sum())
    report["appointment_day_min"] = str(d["appointment_day"].min().date())
    report["appointment_day_max"] = str(d["appointment_day"].max().date())
    all_days = pd.date_range(d["appointment_day"].min().normalize(), d["appointment_day"].max().normalize())
    present = set(d["appointment_day"].dt.normalize())
    report["dates_without_appointments"] = [f"{x.date()} ({x.day_name()})" for x in all_days if x not in present]
    return d.reset_index(drop=True), report


def chronological_split(df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2):
    """Bagi menurut tanggal janji temu; semua baris satu hari masuk satu bagian."""
    day = df["appointment_day"].dt.normalize()
    daily = day.groupby(day).size().sort_index()
    cum = daily.cumsum() / daily.sum()
    train_end = cum.index[int(np.argmax((cum >= train_frac).values))]
    val_end = cum.index[int(np.argmax((cum >= train_frac + val_frac).values))]
    train = df[day <= train_end]
    val = df[(day > train_end) & (day <= val_end)]
    test = df[day > val_end]
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError(
            "Salah satu split kosong; data terlalu terkonsentrasi pada sedikit hari. "
            "Ubah train_frac/val_frac atau periksa rentang tanggal."
        )
    return train.copy(), val.copy(), test.copy()


def drop_seen_patients(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    """Sensitivitas: buang baris val/test dari pasien yang sudah muncul di split sebelumnya."""
    train_ids = set(train["patient_id"])
    val_new = val[~val["patient_id"].isin(train_ids)].copy()
    seen_before_test = train_ids | set(val["patient_id"])
    test_new = test[~test["patient_id"].isin(seen_before_test)].copy()
    return val_new, test_new


def feature_columns(exclude_sms: bool = False) -> list[str]:
    binary = [c for c in BINARY if not (exclude_sms and c == "sms_received")]
    return NUMERIC + binary + CATEGORICAL


def build_preprocessor(exclude_sms: bool = False) -> ColumnTransformer:
    binary = [c for c in BINARY if not (exclude_sms and c == "sms_received")]
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("bin", "passthrough", binary),
            (
                "cat",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50, sparse_output=False),
                CATEGORICAL,
            ),
        ]
    )
