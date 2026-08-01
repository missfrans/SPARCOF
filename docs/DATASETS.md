# Dataset provenance and experimental-use record

This record identifies the exact dataset products expected by the paper
configurations, how their labels are interpreted, and which columns are removed.
Raw datasets are not redistributed in this repository. Provider metadata below
was checked on **2026-08-01**.

## Processing shared by the paper configurations

- Random seed: `42`.
- Final holdout: stratified 80/20 split (`test_size: 0.2`).
- Development evaluation: 10-fold stratified cross-validation on the 80% split.
- Dataset-level row sampling: none in the full paper runs. The optional CLI
  argument `--max-rows-per-dataset` is intended for validation/smoke runs and
  must not be used when producing the reported full-data results.
- Feature-selection sampling: when the 80% development partition exceeds
  100,000 rows, feature selectors use a deterministic stratified sample of at
  most 100,000 rows (`max_rows_for_fs: 100000`). This cap does not replace the
  final 20% holdout.
- Runtime `dataset_fingerprint` values hash file identity, size, and modification
  time for resume protection. They are **not byte-level content checksums** and
  do not replace provider checksums or locally recorded SHA-256 values.

## UNSW-NB15

- **Provider and canonical URL:** UNSW Canberra, [The UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset).
- **Version/release date:** introduced with the 2015 UNSW-NB15 publication. The
  provider does not assign a version identifier to the downloadable CSV release;
  its landing page was last updated on 2021-06-02.
- **Dataset profile:** 2,540,044 records distributed across four CSV files, with
  nine attack families: Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic,
  Reconnaissance, Shellcode, and Worms. The provider documents 49 fields including
  the class label.
- **Date accessed/downloaded:** 2026-01-03
- **License/terms:** free use in perpetuity for academic research, with citation
  of the provider-designated papers; commercial use requires agreement from the
  authors. Check the canonical page again before redistribution. This repository
  does not redistribute the raw data.
- **Files used by this pipeline:**
  - `data/raw/UNSW-NB15/CSV_Files/UNSW-NB15_1.csv`
  - `data/raw/UNSW-NB15/CSV_Files/UNSW-NB15_2.csv`
  - `data/raw/UNSW-NB15/CSV_Files/UNSW-NB15_3.csv`
  - `data/raw/UNSW-NB15/CSV_Files/UNSW-NB15_4.csv`
  - the provider's column-description file `UNSW-NB15_features.csv`; the legacy
    code/archive may name the same local file `NUSW-NB15_features.csv`.
  - The provider's pre-made `UNSW_NB15_training-set.csv` and
    `UNSW_NB15_testing-set.csv` are **not** used; SPARCOF concatenates the four
    full CSV files and creates its own deterministic split.
- **Provider checksum(s):** The provider does not publish checksums on the canonical landing page. Local SHA-256 checksums are recorded in [`dataset_sha256.csv`](dataset_sha256.csv).
- **Target definition and mapping:** binary classification using `label`, where
  `0 = Normal/Benign` and `1 = Attack`. `positive_label: 1` identifies the attack
  class as positive.
- **Excluded identifier/leakage columns and rationale:** `attack_cat` is removed
  because it is an alternative attack-type label and would leak target
  information into the binary task. `srcip`, `sport`, `dstip`, and `dsport` are
  removed as endpoint identifiers/high-cardinality identifiers that may encode
  capture-specific hosts and reduce cross-environment validity.
- **Sampling performed:** no dataset-level row downsampling in the full run.
  The shared 100,000-row feature-selection cap and 80/20 split described above
  apply.

## HIKARI-2021

- **Provider and canonical URL:** Ferriyan et al./Keio University, Zenodo
  [version 1.3.5](https://doi.org/10.5281/zenodo.5199540). Dataset design and label
  definitions are documented in the [peer-reviewed HIKARI-2021 paper](https://doi.org/10.3390/app11177868).
- **Version/release date:** `1.3.5`, published 2021-06-01; version-specific DOI
  `10.5281/zenodo.5199540`.
- **Dataset profile:** 555,278 flows with 86 features. The paper reports two
  benign traffic categories (`Background`, `Benign`) and four attack categories
  (`Bruteforce`, `Bruteforce-XML`, `Probing`, and `XMRIGCC CryptoMiner`).
- **Date accessed/downloaded:** 2026-01-03
- **License/terms:** The HIKARI-2021 Zenodo v1 record states CC BY 4.0. Version 1.3.5 was used in this study. Raw dataset files are not redistributed; researchers must obtain them from the version-specific Zenodo record.
- **Files used by this pipeline:** local file
  `data/raw/HIKARI2021/HIKARI2021.csv`, which is the study's locally renamed
  and extracted copy of the provider file `ALLFLOWMETER_HIKARI2021.csv`.
  The generic loader reads and concatenates every `*.csv` in this directory,
  so no unrelated CSV file may be stored there. The PKL, PCAP, and
  ground-truth archives are not read by SPARCOF.
- **Provider checksum(s):** Zenodo MD5 for `ALLFLOWMETER_HIKARI2021.csv.zip`: `d7d9e277fe4a66cb00764d7f91a810dd`. Local SHA-256 checksums are recorded in [`dataset_sha256.csv`](dataset_sha256.csv).
- **Target definition and mapping:** binary classification using `Label`, where
  the official definition is `0 = Benign` and `1 = Attack`. The categorical
  `traffic_category` column is not used as the target. Consequently, the paper
  configurations specify `task: binary`.
- **Excluded identifier/leakage columns and rationale:** `traffic_category` is
  removed because it is an alternative fine-grained label and would leak the
  binary target. `uid`, `originh`, `originp`, `responh`, and `responp` are removed
  as flow/endpoint identifiers. `no` and `Unnamed: 0` are removed as exported row
  indices rather than traffic measurements.
- **Sampling performed:** no dataset-level row downsampling in the full run.
  The shared 100,000-row feature-selection cap and 80/20 split described above
  apply.

## CICIoT-2023

- **Provider and canonical URL:** Canadian Institute for Cybersecurity,
  University of New Brunswick, [CIC IoT Dataset 2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html).
  The associated paper is [CICIoT2023: A Real-Time Dataset and Benchmark for
  Large-Scale Attacks in IoT Environment](https://doi.org/10.3390/s23135941).
- **Version/release date:** 2023. No formal version identifier or original archive name was retained; file-level SHA-256 checksums identify the dataset snapshot used.
- **Dataset profile:** traffic from a topology of 105 IoT devices, covering 33
  attacks grouped into seven families: DDoS, DoS, Recon, Web-based, Brute Force,
  Spoofing, and Mirai. The provider supplies PCAP, extracted CSV, example, and
  supplementary-material directories; SPARCOF uses only the extracted CSV data.
- **Date accessed/downloaded:** 2026-01-03
- **License/terms:** The official UNB/CIC landing page does not state an explicit redistribution license. Raw dataset files are therefore not included in this repository. Researchers must obtain the dataset directly from UNB/CIC and comply with the applicable download terms.
- **Files used by this pipeline:** five local analysis partitions:
  `Merged01.csv`, `Merged02.csv`, `Merged03.csv`, `Merged04.csv`, and
  `Merged05.csv`, stored under `data/raw/CICIoT2023/dataset/`. The files are
  sorted by filename and concatenated by the pipeline. PCAP files are not read.
- **Provider checksum(s):** The provider does not publish checksums on the canonical landing page. Local SHA-256 checksums are recorded in [`dataset_sha256.csv`](dataset_sha256.csv).
- **Target definition and mapping:** binary classification using `Label`. Values
  equal to `Benign` (case-insensitive) or `0` map to `Benign`; every other raw
  label maps to `Attack`. This collapses the provider's individual attacks and
  seven attack families into one attack class.
- **Excluded identifier/leakage columns and rationale:** none are explicitly
  dropped by the current CICIoT configuration. The final paper should state this
  fact. If a locally downloaded release contains extra identifiers not present
  in the provider's standard feature CSVs, declare them in `drop_columns`,
  `identifier_columns`, or `leakage_columns` before running the experiment.
- **Sampling performed:** files are concatenated and rows are deterministically
  shuffled with seed `42`; no dataset-level row downsampling occurs in the full
  run. The shared 100,000-row feature-selection cap and 80/20 split described
  above apply.

## Create the local checksum manifest

Run this after placing the exact research files under `data/raw/`. Save the
result as a release asset or under `docs/` only if it contains no private paths.

Linux/macOS:

```bash
find data/raw -type f -print0 | sort -z | xargs -0 sha256sum > dataset_sha256.txt
```

Windows PowerShell (run from the repository root):

```powershell
Get-ChildItem data/raw -Recurse -File |
  Sort-Object FullName |
  Get-FileHash -Algorithm SHA256 |
  Select-Object Hash, Path |
  Export-Csv dataset_sha256.csv -NoTypeInformation
```

Review the manifest before publication: paths should be repository-relative and
must not expose a username, drive layout, network share, or other private system
information.
