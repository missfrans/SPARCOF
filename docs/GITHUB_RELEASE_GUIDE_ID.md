# Panduan Langkah demi Langkah Publikasi GitHub

## Tahap A — tetapkan identitas ilmiah

1. Tentukan nama repository yang stabil, misalnya `sparcof-reproducible`.
2. Lengkapi `CITATION.cff.template`, kemudian ubah namanya menjadi
   `CITATION.cff`.
3. Pilih lisensi bersama pemilik kode/institusi. Jangan menerbitkan placeholder
   lisensi sebagai lisensi final.
4. Lengkapi provenance seluruh dataset dalam `docs/DATASETS.md`.
5. Ganti `[GitHub URL]`, DOI, judul, dan commit/release tag pada pernyataan Code
   Availability.

## Tahap B — siapkan environment bersih

```bash
git clone <URL-REPOSITORY-BARU>
cd sparcof-reproducible
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
python scripts/make_demo_dataset.py
sparcof --config configs/example.yaml --validate-only
pytest
sparcof --config configs/example.yaml --mode smoke --force
```

Jika langkah ini gagal pada clone bersih, repository belum siap dirujuk dalam
artikel.

## Tahap C — buat konfigurasi penelitian final

1. Gunakan `configs/paper/full_revision_config.yaml` sebagai salinan protokol
   artikel, bukan sebagai tempat eksperimen coba-coba.
2. Pastikan semua path relatif terhadap root repository.
3. Catat kolom identifier, leakage, label mapping, seed, jumlah fold, test size,
   classifier, selector, dan bobot score.
4. Jalankan `--validate-only`, lalu smoke, focused, dan full.
5. Jangan gunakan `--resume` setelah config atau dataset berubah; gunakan output
   directory baru atau `--force`.
6. Simpan environment final:

```bash
python --version > environment-python.txt
python -m pip freeze > requirements-lock.txt
```

Untuk GPU, tambahkan output `nvidia-smi` dan versi CUDA/cuDNN/PyTorch ke metadata
release.

## Tahap D — pemeriksaan sebelum commit

```bash
git status --short
git diff --check
python -m compileall -q src tests
pytest
```

Periksa bahwa tidak ada data mentah, credential, `.env`, model besar, path lokal,
cache, atau identitas sensitif:

```bash
git ls-files | grep -E '(^data/raw/|\.env$|__pycache__|\.pyc$|\.pkl$|\.pt$)'
git grep -n -E '(api[_-]?key|password|secret|token)' -- ':!docs/*'
```

Hasil kosong pada dua pencarian terakhir adalah kondisi yang diharapkan; temuan
harus diperiksa manual, bukan dihapus membabi buta.

## Tahap E — commit dan push

```bash
git init
git branch -M main
git add .
git commit -m "Release reproducible SPARCOF research code"
git remote add origin https://github.com/<OWNER>/<REPOSITORY>.git
git push -u origin main
```

Di GitHub, aktifkan Issues bila akan menerima laporan reproduksi, tambahkan topic
`reproducible-research`, `feature-selection`, dan bidang riset yang relevan.

## Tahap F — release yang dapat disitasi

1. Buat tag, misalnya `v1.0.0-paper`.
2. Buat GitHub Release dan tuliskan config final, checksum/versi dataset,
   environment, hardware, serta hubungan release dengan versi manuskrip.
3. Bila memakai Zenodo, hubungkan repository lalu mintakan DOI untuk release.
4. Gunakan URL release/tag atau DOI arsip dalam artikel, bukan hanya URL branch
   `main` yang dapat berubah.
5. Setelah artikel diterima, perbarui DOI artikel pada citation metadata melalui
   commit dan release baru; jangan mengubah diam-diam release yang sudah disitasi.
