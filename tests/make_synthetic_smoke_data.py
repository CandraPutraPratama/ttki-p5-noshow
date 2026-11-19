"""DATA TIRUAN hanya untuk smoke test kode (bukan data Kaggle, bukan hasil penelitian).

Membuat CSV dengan skema yang sama dengan dataset Kaggle supaya pipeline bisa
diuji tanpa mengunduh data. Angka yang keluar dari data ini TIDAK BOLEH
dilaporkan di paper.
"""
import sys

import numpy as np
import pandas as pd

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/synthetic_smoke.csv"
rng = np.random.default_rng(123)
n = 30000
appt = pd.Timestamp("2016-04-29") + pd.to_timedelta(rng.integers(0, 40, n), unit="D")
lead = rng.integers(0, 40, n)
sched = appt - pd.to_timedelta(lead, unit="D") + pd.to_timedelta(rng.integers(7 * 3600, 18 * 3600, n), unit="s")
age = rng.integers(0, 95, n)
sms = ((lead > 2) & (rng.random(n) < 0.6)).astype(int)
logit = -1.9 + 0.03 * lead - 0.01 * age + 0.2 * sms + rng.normal(0, 0.5, n)
p = 1 / (1 + np.exp(-logit))
no_show = np.where(rng.random(n) < p, "Yes", "No")
df = pd.DataFrame(
    {
        "PatientId": rng.integers(1, 12000, n).astype(float),
        "AppointmentID": np.arange(5_000_000, 5_000_000 + n),
        "Gender": rng.choice(["F", "M"], n),
        "ScheduledDay": sched.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "AppointmentDay": appt.strftime("%Y-%m-%dT00:00:00Z"),
        "Age": age,
        "Neighbourhood": rng.choice([f"AREA_{i}" for i in range(30)], n),
        "Scholarship": rng.integers(0, 2, n),
        "Hipertension": rng.integers(0, 2, n),
        "Diabetes": rng.integers(0, 2, n),
        "Alcoholism": rng.integers(0, 2, n),
        "Handcap": rng.integers(0, 3, n),
        "SMS_received": sms,
        "No-show": no_show,
    }
)
df.to_csv(out, index=False)
print("tersimpan:", out, df.shape)
