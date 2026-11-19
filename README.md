# TTKI P5: Prediksi Ketidakhadiran Pasien untuk Prioritas Pengingat

Repositori ini memuat kode untuk proyek akhir Tata Tulis Karya Ilmiah (PENS, semester 5, 2026), topik P5.

**Research question:** Apakah optimasi threshold dan kalibrasi probabilitas meningkatkan kualitas prioritisasi pasien berisiko no-show pada kapasitas pengingat 10%?

**Status:** eksperimen sudah lengkap. Eksperimen utama (`results/`), sensitivitas tanpa SMS (`results_no_sms/`), dan sensitivitas pasien belum pernah terlihat (`results_unseen/`) sudah dijalankan pada CSV Kaggle, beserta gambar di `figures*/`. Run ulang di mesin lain memberi hasil yang sama. Log pencarian gap ada di `references/gap_search_log.csv`.

## Sumber data dan atribusi

- Dataset: Medical Appointment No Shows, Kaggle, <https://www.kaggle.com/datasets/joniarroba/noshowappointments>
- Berkas yang dipakai: `KaggleV2-May-2016.csv` (110.527 baris, 14 kolom), SHA-256 `9132d3e7d0246617df9041d3764f20ad6f08e7b0d9f0997fa254fc5e52eda27d`.
- Lisensi: CC BY-NC-SA 4.0, <https://creativecommons.org/licenses/by-nc-sa/4.0/> (tautan diberikan anggota kelompok; cocokkan dengan halaman Kaggle). Versi berkas diverifikasi lewat hash SHA-256 di atas.
- Repositori referensi baseline: <https://github.com/Rachel-Oyeyemi/Medical-Appointment-No-Shows> (memakai data sintetis; tidak ada angka dari repositori itu yang dipakai di sini).
- Berkas CSV tidak dikomit. Unduh sendiri dari Kaggle dan taruh di `data/` (lihat `data/README.md`).

## Environment

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m pip freeze > environment-freeze.txt
```

Dibutuhkan scikit-learn 1.6 atau lebih baru (memakai `FrozenEstimator`). Versi yang dipakai tiap run dicatat otomatis di `results/run_config.json`.

## Urutan menjalankan

```bash
# 1. Audit dataset (catat hasilnya di bagian Methods)
python src/audit_dataset.py --data data/<nama-file>.csv

# 2. Eksperimen utama: baseline, skenario A, B, C
python src/run_experiments.py --data data/<nama-file>.csv --out results --figures figures

# 3. Sensitivitas (opsional, dilaporkan sebagai analisis tambahan)
python src/run_experiments.py --data data/<nama-file>.csv --out results_no_sms --figures figures_no_sms --exclude-sms
python src/run_experiments.py --data data/<nama-file>.csv --out results_unseen --figures figures_unseen --unseen-patients
```

## Rancangan eksperimen

| Komponen | Rancangan |
| --- | --- |
| Unit observasi | Satu janji temu (`AppointmentID`) |
| Target | `No-show` (Yes = 1) |
| Fitur | Usia, jarak hari jadwal ke janji (`lead_days`), jam dan hari penjadwalan, hari janji, jenis kelamin, beasiswa, hipertensi, diabetes, alkoholisme, handicap, `SMS_received`, lingkungan (one-hot, kategori jarang digabung) |
| Fitur terlarang | `PatientId`, `AppointmentID` (identitas, tidak dipakai sebagai fitur) |
| Split | Kronologis menurut `AppointmentDay`: 60% train, 20% validation, 20% test (satu hari penuh selalu di satu bagian) |
| Kebocoran | Scaler dan encoder di-fit hanya pada train (Pipeline). Test tidak dipakai untuk tuning apa pun |
| Seed | Random Forest dijalankan dengan seed 0 sampai 4; Logistic Regression deterministik |

Skenario:

- **baseline**: Logistic Regression dan Random Forest, threshold 0,5.
- **A**: threshold dipilih pada validation set dengan kriteria F1 maksimum, lalu diterapkan ke test.
- **A2** (eksploratif, ditambahkan setelah run pertama): threshold sama dengan persentil ke-90 skor validation, sehingga kira-kira 10% janji temu ditandai tanpa melihat test. Laporkan sebagai analisis tambahan.
- **B**: probabilitas dikalibrasi (sigmoid dan isotonic) pada validation set yang terpisah dari train dan test.
- **C**: capaian pada kapasitas 10% teratas (precision@10%, recall@10%, lift) untuk semua skor, dibandingkan dengan pemilihan acak dan heuristik `lead_days`.

Metrik: PR-AUC, ROC-AUC, precision@10%, recall@10%, lift@10%, Brier score, ECE, serta precision/recall/F1 pada threshold. Interval kepercayaan 95% (bootstrap baris, seed 0) ada di `results/bootstrap_ci.csv`.

## Keluaran

| Berkas | Isi |
| --- | --- |
| `results/audit.json` | Ringkasan audit dan jumlah baris yang dibuang |
| `results/dataset_characteristics.csv` | Ukuran, rentang tanggal, no-show rate, dan tumpang tindih pasien per split |
| `results/metrics.csv` | Metrik per skenario, model, dan seed |
| `results/summary.csv` | Rata-rata dan simpangan baku antar seed |
| `results/bootstrap_ci.csv` | Interval kepercayaan bootstrap |
| `results/run_config.json` | Argumen, versi paket, hash SHA-256 data |
| `figures/` | Kurva PR, reliability diagram, precision@k |

## Keterbatasan yang perlu dilaporkan

- Satu pasien bisa muncul di beberapa split pada split kronologis; tumpang tindihnya tercatat di `dataset_characteristics.csv`, dan sensitivitas `--unseen-patients` menguji dampaknya.
- `SMS_received` bersifat observasional. Jangan menyimpulkan efek kausal SMS terhadap kehadiran.
- Data berasal dari satu kota dan satu periode; generalisasi terbatas.

## Pengujian kode

`tests/make_synthetic_smoke_data.py` membuat data tiruan hanya untuk memastikan kode berjalan. Hasil dari data tiruan tidak boleh dilaporkan.

```bash
python tests/make_synthetic_smoke_data.py /tmp/synthetic_smoke.csv
python src/run_experiments.py --data /tmp/synthetic_smoke.csv --out /tmp/smoke_out --figures /tmp/smoke_fig --seeds 2 --n-boot 50
```

## Integritas akademik

Tidak ada token Kaggle, kata sandi, atau data privat di repositori ini. Jika memakai bantuan AI untuk kode atau tulisan, laporkan sesuai kebijakan kelas.
