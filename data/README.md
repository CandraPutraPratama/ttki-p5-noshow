# Data

CSV tidak disimpan di repositori. Cara mengambilnya:

1. Buka <https://www.kaggle.com/datasets/joniarroba/noshowappointments> (perlu akun Kaggle) lalu unduh dataset.
2. Ekstrak dan letakkan CSV di folder ini.
3. Catat di README utama: tanggal unduh, versi dataset, dan lisensi yang tertulis di halaman Kaggle.

Alternatif dengan Kaggle CLI memakai token milik sendiri (jangan pernah mengomit `kaggle.json`):

```bash
kaggle datasets download -d joniarroba/noshowappointments -p data --unzip
```

Kolom yang diharapkan: PatientId, AppointmentID, Gender, ScheduledDay, AppointmentDay, Age, Neighbourhood, Scholarship, Hipertension, Diabetes, Alcoholism, Handcap, SMS_received, No-show. Kode menormalkan variasi nama kolom, dan berhenti dengan pesan jelas jika ada kolom wajib yang tidak ada.
