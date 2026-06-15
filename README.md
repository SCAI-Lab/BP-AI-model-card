# BP-AI-model-card
Blood Pressure Estimation Models: A Guideline and Tutorial on Correct Model reporting for developing estimation models that learn from secondary/indirect vital signs to estimate blood pressure. 

# BP-Card

A reporting and validation framework for AI-driven cuffless blood pressure (BP) estimation.

BP-Card standardizes how cuffless BP models are reported and validated, so results are
comparable across studies. It is the reporting instrument from *Towards Trustworthy
AI-driven Cuffless Blood Pressure Monitoring* (Cisnal et al., npj Digital Medicine, in
press). This repository provides a small, dependency-light Python implementation.


<p align="center">
  <img src="figures/bpcard_framework.pdf"
       alt="Overview of the BP-Card reporting and validation framework"
       width="800">
</p>

<p align="center">
  <em>Figure 1. Overview of the BP-Card reporting and validation framework.</em>
</p>

## What This Repository Provides

- Standardized BP-Card performance tables (Modules 3A / 3B)
- Metrics: MD±SD, MAD, MAPD, CP5 / CP10 / CP15, R, and ΔBP±SD
- AAMI, BHS, and IEEE 1708 conformance-grading helpers
- Correlation and Bland–Altman plotting utilities
- A reporting checklist for BP-Card Modules 0–3

## Quick Start

```bash
pip install -r requirements.txt
python bpcard_metrics.py        # runs a self-contained demonstration
```

## Minimal Example

```python
import pandas as pd
from bpcard_metrics import bp_category, performance_table

df = pd.read_csv("examples/sample_results.csv")     # ref_sbp, pred_sbp, calib_sbp, ...
df["group"] = [bp_category(s) for s in df["ref_sbp"]]

table = performance_table(
    df, reference="ref_sbp", prediction="pred_sbp",
    group="group",          # BP category (3A) or calibration condition (3B)
    baseline="calib_sbp",   # adds the ΔBP±SD column; omit if not applicable
    grades=True,            # append AAMI / BHS / IEEE conformance columns
)
print(table.to_string())
print(table.to_latex())     # publication-ready
```

A complete runnable script is in [`examples/minimal_example.py`](examples/minimal_example.py).

## BP-Card Framework

The framework has four reporting modules; Module 3 is reported separately for each
calibration pathway:

- **Module 0** — clinical purpose and intended use
- **Module 1** — dataset and cohort characteristics (including BP and ΔBP distributions)
- **Module 2** — validation strategy (subject-wise validation is the minimum standard)
- **Module 3A / 3B** — standardized performance reporting for calibration-free / calibration-based models

See [`docs/framework.md`](docs/framework.md) for the modules and the paper's core
methodological requirements.

## Metrics and Standards

The recommended report set is **MD±SD, CP5, CP10, CP15, MAD, MAPD, and R** (MAD±SD is
discouraged as redundant). Conformance is graded against AAMI/ISO, BHS (A–D), and
IEEE 1708 (A–D).

> **ΔBP sign convention.** ΔBP±SD describes the BP variation in the evaluation set, not
> an error. Its sign depends on the subtraction order. This code defaults to the paper's
> worked-example convention, `ΔBP = baseline − current` (`delta_convention=
> "baseline_minus_current"`). The SD is identical either way; only the mean's sign
> changes. **State your convention when you report ΔBP.**

Full definitions, formulas, and per-metric code are in [`docs/metrics.md`](docs/metrics.md).

## Reporting Checklist

<details>
<summary>BP-Card Modules 0–2 checklist (click to expand)</summary>

**Module 0 — Clinical purpose and use context**
- [ ] Model name and version, estimation family, calibration paradigm
- [ ] Model type, input data type, output variable (SBP/DBP/MAP)
- [ ] Signal modality, sensor specification and location
- [ ] Intended use designation

**Module 1 — Dataset and cohort reporting**
- [ ] Source, cohort size, recording sessions, total paired samples
- [ ] Demographics (age, sex, BMI, height, weight, ethnicity)
- [ ] Comorbidities, medication (class/dose/regimen), known sensitivities; skin tone and ambient light for PPG
- [ ] Measurement phases and BP profile distributions
- [ ] SBP/DBP, baseline SBP/DBP, and ΔSBP/ΔDBP (subject-specific or population-based)
- [ ] Preprocessing pipeline, exclusion criteria, excluded sample count
- [ ] BP distribution reported before and after cleaning

**Module 2 — Validation strategy**
- [ ] Evaluation method (subject-wise required as minimum standard)
- [ ] Unseen-subject test set and external validation
- [ ] Reference standard and device (manufacturer/model)
- [ ] Measurement conditions, body position, activity state, rest period
- [ ] Baseline comparator and result
- [ ] Modules 3A and 3B reported separately

</details>

The full checklist, including Module 3, is in [`docs/reporting-checklist.md`](docs/reporting-checklist.md).

## Citation

If you use this software or the BP-Card framework, please cite the paper (see also
[`CITATION.cff`](CITATION.cff)):

```bibtex
@article{cisnal2026bpcard,
  title   = {Towards Trustworthy AI-driven Cuffless Blood Pressure Monitoring},
  author  = {Cisnal, Ana and Podder, Itilekha and Grossmann, Leoni and
             Dheman, Kanika and Elgendi, Mohamed and Valgimigli, Marco and
             Paez-Granados, Diego},
  journal = {npj Digital Medicine},
  year    = {2026},
  note    = {Accepted; in press},
  doi     = {<to be assigned>}
}
```

Update the volume, pages, and DOI once assigned.

## Repository Contents

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── CITATION.cff
├── bpcard_metrics.py          # metrics, grading, table, and plots
├── docs/
│   ├── framework.md           # the four modules + core requirements
│   ├── metrics.md             # metric definitions, formulas, code
│   └── reporting-checklist.md # full Modules 0–3 checklist
└── examples/
    ├── minimal_example.py     # runnable end-to-end example
    └── sample_results.csv     # small synthetic dataset
```

## Planned Additions

- Volume / pages / DOI once assigned by *npj Digital Medicine*
- Worked examples for DBP and MAP
- Module 3B calibration-condition example (static / dynamic / time-after-calibration)
- Bootstrap 95% confidence intervals for metrics and Bland–Altman limits of agreement

## License

Released under the MIT License — see [`LICENSE`](LICENSE). Confirm the license and
copyright holder suit your group before publishing.
