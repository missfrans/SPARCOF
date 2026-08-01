# Laporan Audit Teknis dan Reproduksibilitas

## Kesimpulan

Arsip awal sudah lebih baik daripada kumpulan skrip penelitian biasa: terdapat
seed, indeks split, checkpoint, konfigurasi, output rinci, dan pemisahan modul.
Namun, dalam bentuk semula repository belum aman disebut dataset-agnostic dan
belum sepenuhnya siap dipublikasikan. Perbaikan di repository ini mengatasi
masalah implementasi utama tanpa menghapus konfigurasi serta hasil rujukan tiga
dataset penelitian.

## Temuan kritis

| Temuan | Dampak | Tindakan |
|---|---|---|
| Nama fitur mengikuti urutan kolom mentah, sedangkan matriks preprocessing disusun sebagai seluruh fitur numerik lalu kategorikal | Pada data dengan tipe kolom berselang-seling, indeks fitur terpilih dapat menunjuk kolom transformasi yang salah | Diperbaiki: `feature_names` sekarang selalu mengikuti urutan matriks numerik + kategorikal; diuji otomatis |
| XGBoost/LightGBM dapat gagal lalu diam-diam diganti Random Forest dengan nama metode tetap XGBoost/LightGBM | Salah atribusi algoritme dan sulit direplikasi | Diperbaiki: kegagalan dependency/runtime dicatat sebagai selector gagal, tanpa substitusi tersembunyi |
| Pemilihan Pareto/champion pada konfigurasi lama menggunakan metrik final test untuk semua kandidat | Test set menjadi bagian pemilihan model; estimasi performa champion berisiko optimistis | Untuk penelitian baru tersedia `scoring.metric_source: cross_validation`; konfigurasi artikel lama dipertahankan demi parity dan harus dinyatakan sebagai keterbatasan |

Konsekuensi terpenting: karena perbaikan urutan fitur dapat mengubah fitur yang
benar-benar masuk ke model, hasil artikel sebaiknya direrun sebelum repository
ditautkan sebagai kode final. Tabel lama di `results/paper/` adalah hasil rujukan,
bukan bukti bahwa versi refactor menghasilkan angka identik.

## Temuan tinggi yang telah diperbaiki

1. Loader dan path tiga dataset ditanam langsung dalam kode runner.
2. Konfigurasi per mesin/dataset menggandakan hampir seluruh YAML.
3. Resume tidak memeriksa perubahan config atau data mentah.
4. Nama classifier berakhiran `_gpu` dapat berjalan pada CPU tanpa backend aktual
   disimpan dalam hasil.
5. Nilai kategorikal hilang diubah menjadi string `"nan"` sebelum imputer, sehingga
   tidak benar-benar diimputasi.
6. Sampling cepat dan sampling feature selection tidak distratifikasi.
7. Folder `__pycache__` dan bytecode beberapa versi Python ikut dipublikasikan.
8. Tidak tersedia tes regresi, package metadata, dan command-line entry point
   yang dapat diinstal.
9. Path relatif bergantung pada current working directory; sekarang ditambatkan
   pada root repository.

## Catatan metodologis yang belum boleh disembunyikan

- Feature selection dilakukan pada development/train-validation set secara
  keseluruhan sebelum metrik fold dihitung. Karena itu, metrik CV berfungsi untuk
  pemilihan internal, bukan estimasi performa eksternal yang sepenuhnya nested.
  Holdout test tetap menjadi ukuran akhir apabila champion dipilih hanya dari CV
  dan keputusan dikunci sebelum membaca metrik test.
- Pipeline mengevaluasi holdout untuk seluruh kandidat demi kompatibilitas hasil
  lama. Pada studi baru, peneliti harus mengunci aturan pemilihan berbasis CV dan
  hanya menafsirkan holdout champion yang telah dipilih.
- Timing dipengaruhi hardware, OS, versi driver, beban mesin, backend, dan ukuran
  batch. Nilai lintas mesin tidak boleh dibandingkan tanpa metadata lingkungan.
- `TP_sum`, `FP_sum`, `TN_sum`, dan `FN_sum` pada kode lama merupakan agregasi
  one-vs-rest lintas kelas; pada multiclass tidak identik dengan satu confusion
  matrix biner. Untuk artikel, utamakan confusion matrix per kelas, macro DR/FAR,
  dan metrik weighted yang definisinya eksplisit.
- Rentang dependency di `pyproject.toml` memudahkan instalasi, tetapi reproduksi
  jangka panjang tetap membutuhkan lock file atau export environment dari mesin
  yang menghasilkan angka final.

## Status keputusan publikasi

- **Lisensi kode:** telah ditetapkan menggunakan MIT License sebagaimana
  tercantum dalam file `LICENSE`. Lisensi ini hanya berlaku untuk kode dan
  dokumentasi repository, bukan untuk dataset penelitian.
- **Metadata sitasi:** telah disediakan melalui `CITATION.cff`, termasuk judul
  perangkat lunak, versi, tanggal rilis, penulis, URL repository, dan preferred
  article citation. DOI artikel akan ditambahkan setelah diterbitkan oleh jurnal.
- **Provenance dataset:** sumber resmi, tanggal akses, ketentuan penggunaan,
  file yang digunakan, pemetaan target, kolom yang dikeluarkan, dan prosedur
  sampling telah didokumentasikan dalam `docs/DATASETS.md`.
- **Checksum dataset:** SHA-256 dan ukuran setiap file yang digunakan telah
  dicatat dalam `docs/dataset_sha256.csv`. Dataset mentah tidak didistribusikan
  dalam repository.
- **Keputusan metodologis yang tetap harus konsisten dengan artikel:** penulis
  harus memastikan bahwa penggunaan `scoring.metric_source` pada konfigurasi,
  hasil yang dilaporkan, manuscript, dan response-to-reviewer merujuk pada
  protokol yang sama. Hasil dari protokol `final_test` tidak boleh dicampur
  dengan hasil dari protokol `cross_validation` tanpa penjelasan eksplisit.
- **Verifikasi hasil sebelum release:** tabel dalam `results/paper/` harus
  dipastikan berasal dari hasil rerun yang digunakan dalam versi akhir
  manuscript. Jika tabel tersebut hanya merupakan hasil rujukan lama, statusnya
  harus tetap dinyatakan secara jelas.
