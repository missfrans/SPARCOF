# Results

- `paper/` contains the supplied reference tables from the authors' completed
  rerun. These are evidence artifacts, not inputs to the pipeline. Their
  integrity manifest is `paper/SHA256SUMS`.
- `outputs/` is generated locally and ignored by Git.

Never mix outputs from different configurations in one output directory. Resume
guards reject changed configs and changed dataset fingerprints.
