# BP-Card Framework

BP-Card is a model-card-inspired reporting framework for AI-driven cuffless blood
pressure (BP) estimation, introduced in *Towards Trustworthy AI-driven Cuffless Blood
Pressure Monitoring* (Cisnal et al., npj Digital Medicine, in press). It specifies
minimum reporting requirements and separates the calibration-free and calibration-based
evaluation pathways.

```html
<p align="center">
  <img src="figures/figure-9-1.png"
       alt="Worked example of the BP-Card reporting framework applied to a calibration-based cuffless blood pressure model using the Aurora-BP dataset."
       width="800">
</p>

<p align="center">
  <em>
  Worked example of the proposed BP-Card reporting framework applied to a calibration-based cuffless BP model using the Aurora-BP dataset. The example reports the clinical purpose and use context (Module 0), dataset and cohort characteristics (Module 1), validation strategy (Module 2), and calibration-specific performance evaluation (Module 3B). The visualizations include train and test ΔSBP distributions, ΔSBP change from calibration, scatter plots, and Bland–Altman agreement analysis.
  </em>
</p>
```



## Why it exists

The paper's systematic review and meta-analysis of 86 cuffless BP studies (2017–2025)
found that the field's main limitation is inconsistent, underpowered, and
methodologically heterogeneous reporting rather than selective publication:

- Pooled SBP mean error was small (1.12 mmHg, 95% CI [0.76, 1.47]) but extremely
  heterogeneous (I² = 98.8%).
- Subject-wise studies showed a smaller, more stable bias (μ ≈ 0.81 mmHg) than
  record-wise studies (μ ≈ 1.50 mmHg) with wider dispersion — consistent with
  information leakage.
- No study achieved low overall risk of bias; validation independence was the weakest
  domain.
- None of the reviewed studies reported a baseline-model comparison.

BP-Card translates these findings into structured reporting requirements.

## The four modules

| Module | Scope | Rationale |
|---|---|---|
| **0 — Clinical purpose and use context** | Model name and version, estimation family, calibration paradigm, model type, input data type, output variable (SBP/DBP/MAP), signal modality, sensor specification and location, intended use | Operationalizes the first two taxonomy axes. The intended-use designation (research, home/ambulatory, adjunctive clinical, clinical pilot) sets the evidentiary bar for everything that follows. |
| **1 — Dataset and cohort reporting** | Source (public/private), cohort size, demographics, comorbidities and medication, measurement phases, BP and ΔBP distributions, preprocessing and exclusions, distributions before/after cleaning | Cohort composition and BP variability drive apparent performance. Narrow BP distributions can produce favorable error metrics that do not reflect predictive capability. |
| **2 — Validation strategy** | Evaluation method, unseen-subject and external validation, reference standard and device, measurement conditions, baseline comparator | Validation design determines whether performance generalizes. Subject-wise partitioning with an unseen-subject test set is the central requirement; the baseline comparison contextualizes the signal's contribution. |
| **3A / 3B — Performance reporting** | Standardized metric table and recommended visualizations, stratified by BP category (3A) or calibration condition (3B), with baseline results in both | A common format enables direct comparison and prevents selective metric reporting. |

## Core methodological requirements

- **Subject-wise validation as the minimum acceptable standard.** Record-, window-, or
  sample-wise splits allow the same individual to appear in training and test sets,
  inflating apparent accuracy.
- **Calibration-free and calibration-based pathways reported separately.** The
  calibration axis determines the appropriate validation and reporting logic.
- **Baseline-model comparison in both pathways.** Calibration-based baselines use each
  subject's calibration BP; calibration-free baselines use population-average BP (and
  demographics where applicable), separating genuine signal contribution from
  population priors.
- **Pathway-specific evaluation conditions.**
  - *Calibration-free:* a broad BP spectrum. IEEE 1708a-2019 requires ≥ 85 subjects with
    ~21 each across normotension, prehypertension, stage 1, and stage 2.
  - *Calibration-based:* static **and** dynamic testing, plus assessment of time since
    calibration. IEEE 1708a recommends inducing BP changes within [−30, 30] mmHg (SBP)
    and [−20, 20] mmHg (DBP), with at least `n_total = 255 + 3·(n − 85)` accuracy data
    points (see `ieee_min_sample_size` in `bpcard_metrics.py`).

