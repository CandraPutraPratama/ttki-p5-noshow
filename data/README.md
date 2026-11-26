# Data

CSV tidak disimpan di repositori.

## Metadata dataset

- Nama: Medical Appointment No Shows (pemilik di Kaggle: joniarroba)
- Tautan: <https://www.kaggle.com/datasets/joniarroba/noshowappointments>
- Berkas: `KaggleV2-May-2016.csv`, 110.527 baris, 14 kolom
- Tanggal unduh: 19 November 2026
- Versi: halaman Kaggle tidak menampilkan nomor versi pada bagian yang diperiksa, jadi berkas diidentifikasi lewat SHA-256
- SHA-256: `9132d3e7d0246617df9041d3764f20ad6f08e7b0d9f0997fa254fc5e52eda27d`
- Lisensi: CC BY-NC-SA 4.0, <https://creativecommons.org/licenses/by-nc-sa/4.0/> (dicek di halaman Kaggle, 19 November 2026)

## Cara mengambil data

1. Buka <https://www.kaggle.com/datasets/joniarroba/noshowappointments> (perlu akun Kaggle) lalu unduh dataset.
2. Ekstrak dan letakkan CSV di folder ini.
3. Pastikan SHA-256 berkas sama dengan yang tercatat di atas.

Alternatif dengan Kaggle CLI memakai token milik sendiri (jangan pernah mengomit `kaggle.json`):

```bash
kaggle datasets download -d joniarroba/noshowappointments -p data --unzip
```

Kolom yang diharapkan: PatientId, AppointmentID, Gender, ScheduledDay, AppointmentDay, Age, Neighbourhood, Scholarship, Hipertension, Diabetes, Alcoholism, Handcap, SMS_received, No-show. Kode menormalkan variasi nama kolom, dan berhenti dengan pesan jelas jika ada kolom wajib yang tidak ada.
