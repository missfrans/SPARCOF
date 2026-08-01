# Code Availability statement template

The source code and configuration files used for this study are archived at
**[persistent release URL or DOI]**, release **[tag]**, commit **[full commit
SHA]**. The archive includes preprocessing, feature-selection, split generation,
model evaluation, repeated timing, effectiveness–efficiency scoring, Pareto
analysis, sensitivity analysis, core-feature coverage, and manuscript-table
generation. Raw datasets are not redistributed because **[license/size/privacy
reason]**; authoritative access locations, dataset versions, checksums, and the
required directory layout are provided in `docs/DATASETS.md`. The exact article
experiment is configured in **[config path]** and can be launched with:

```bash
sparcof --config [config path] --mode full --gpu-required --force
```

Environment and hardware metadata for the archived run are provided in
**[release asset/path]**. Reference outputs are stored under `results/paper/`.
