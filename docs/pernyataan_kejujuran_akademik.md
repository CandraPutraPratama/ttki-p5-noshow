# Pernyataan Kejujuran Akademik

Kami yang bertanda tangan di bawah ini, anggota Kelompok 11 (Topik P5), Program Sarjana Terapan Teknik Informatika, Politeknik Elektronika Negeri Surabaya:

| Nama | NRP |
| --- | --- |
| Candra Putra Pratama | …………… |
| Farrel Athallah Kurniawan | …………… |

menyatakan dengan sebenarnya bahwa paper berjudul **"Threshold Keputusan dan Kalibrasi Probabilitas untuk Prioritisasi Pasien Berisiko No-Show pada Kapasitas Pengingat 10%"** (berkas `TTKI_Kelompok11_TopikP5`) dan repositori <https://github.com/CandraPutraPratama/ttki-p5-noshow> (tag `v1.0-final`) memenuhi butir-butir berikut, sesuai Bab 9.1 Buku Panduan Proyek TTKI.

## 1. Sumber data

- Data berasal dari dataset publik Kaggle *Medical Appointment No Shows* (pemilik: joniarroba), <https://www.kaggle.com/datasets/joniarroba/noshowappointments>, berkas `KaggleV2-May-2016.csv`, diunduh pada 19 November 2026.
- SHA-256 berkas: `9132d3e7d0246617df9041d3764f20ad6f08e7b0d9f0997fa254fc5e52eda27d`.
- Lisensi dataset: CC BY-NC-SA 4.0. Berkas data tidak diunggah ke repositori.
- Tidak ada data privat, token Kaggle API, kata sandi, atau kredensial lain di repositori.

## 2. Hasil eksperimen

- Seluruh angka, tabel, dan gambar hasil di paper berasal dari eksperimen yang kami jalankan sendiri dengan kode di `src/`, dan tersimpan di `results/`, `results_no_sms/`, `results_unseen/`, dan `figures*/`.
- Kami tidak menyalin angka hasil dari README GitHub, paper, atau repositori lain dan mengklaimnya sebagai hasil kami.
- Kami tidak membuat referensi, DOI, tabel, atau gambar yang tidak berasal dari eksperimen atau sumber yang dapat diperiksa. Klaim dan angka dari literatur telah dicocokkan dengan PDF aslinya (lihat `references/literature_matrix.csv`).
- Hasil dilaporkan apa adanya, termasuk hasil yang tidak mendukung harapan (kalibrasi tidak menaikkan kualitas peringkat; threshold berbasis validasi tidak menandai 10% janji temu uji).

## 3. Atribusi

- Repositori referensi baseline pada buku panduan, <https://github.com/Rachel-Oyeyemi/Medical-Appointment-No-Shows>, memakai data sintetis; tidak ada angka dari repositori itu yang kami laporkan.
- Algoritma dan pustaka: scikit-learn (Logistic Regression, Random Forest, kalibrasi), pandas, numpy, dan matplotlib; versi tercatat di `environment-freeze.txt` dan `results/run_config.json`.
- Kode kami berlisensi MIT (`LICENSE`). Karya yang kami rujuk dicantumkan di Daftar Pustaka [1]–[11].

## 4. Penggunaan AI

- Kami menggunakan Claude (Anthropic) untuk membantu menyusun kode eksperimen, analisis, notebook, dan draf tulisan, serta membantu memeriksa kutipan terhadap PDF referensi.
- Seluruh kode dijalankan ulang oleh kami pada komputer sendiri, hasilnya dicocokkan dengan keluaran yang dilaporkan, dan semua isi paper telah kami baca dan verifikasi. Kami bertanggung jawab penuh atas isi paper dan repositori.
- Pernyataan ini juga tercantum pada bagian "Ketersediaan Data dan Kode" paper dan di `README.md`.

## 5. Batas inferensi

- Penelitian ini adalah klasifikasi retrospektif pada data observasional. Kami tidak mengklaim hubungan sebab-akibat, termasuk untuk variabel `SMS_received`.
- Kami tidak mengklaim kebaruan; penelusuran gap kami terbatas dan tercatat di `references/gap_search_log.csv`.

Surabaya, November 2026

| Candra Putra Pratama | Farrel Athallah Kurniawan |
| --- | --- |
| &nbsp; | &nbsp; |
| (tanda tangan) | (tanda tangan) |
| NRP: …………… | NRP: …………… |
