# BP-Card Reporting Checklist (Modules 0–3)

A practical checklist for completing the BP-Card. Modules are filled top to bottom;
Module 3 is reported separately for each calibration pathway.

## Module 0 — Clinical purpose and use context

- [ ] Model name and version
- [ ] Estimation family (hemodynamics-derived / data-driven BP regression / ABP waveform reconstruction)
- [ ] Calibration paradigm (calibration-free / calibration-based)
- [ ] Model type (ML / DL / hybrid / other)
- [ ] Input data type (handcrafted features / time series / images / demographic / other)
- [ ] Output variable (SBP / DBP / MAP)
- [ ] Signal modality (PPG / ECG / BCG / BioZ / IPG / iPPG / other)
- [ ] Sensor specification and location
- [ ] Intended use (research / home or ambulatory / adjunctive clinical / clinical pilot)

## Module 1 — Dataset and cohort reporting

- [ ] Data source (public / private), cohort size, recording sessions, total paired samples
- [ ] Population type (healthy / hypertensive / ICU / other)
- [ ] Demographics (age, sex, BMI, height, weight, ethnicity / region)
- [ ] Medical history / comorbidities
- [ ] Medication (class / dose / regimen)
- [ ] Known sensitivities (motion / low perfusion / arrhythmia / sensor displacement / skin tone)
- [ ] Skin tone / Fitzpatrick and ambient light (PPG-based)
- [ ] Measurement phases (%) and BP profile distributions (%)
- [ ] SBP / DBP (mean ± SD, range)
- [ ] Baseline SBP / DBP (mean ± SD, range)
- [ ] ΔSBP / ΔDBP (mean ± SD; subject-specific or population-based)
- [ ] Preprocessing pipeline
- [ ] Signal quality filtering / exclusion criteria
- [ ] Fiducial points / feature extraction
- [ ] Excluded samples after cleaning
- [ ] BP distribution reported before and after cleaning

## Module 2 — Validation strategy

- [ ] Evaluation method (subject-wise hold-out / subject-wise CV / LOSO / LOOCV / record-wise / sample-wise) — subject-wise is the minimum standard
- [ ] Data split details
- [ ] Unseen-subject test set (yes / no)
- [ ] External validation, independent cohort / site (yes / no)
- [ ] Reference standard and reference device (manufacturer / model)
- [ ] Measurement conditions (resting / ambulatory / dynamic / long-term / ICU)
- [ ] Body position, activity state, rest period prior to measurement
- [ ] Baseline comparator and comparator result
- [ ] Minimum acceptable standard met
- [ ] Pathway-specific analyses reported separately (Module 3A and/or 3B)

## Module 3A — Performance reporting, calibration-free models

- [ ] Models evaluated (baseline model and proposed / final model)
- [ ] Standardized performance table, stratified by BP group (All / Normal / Prehypertension / Stage 1 / Stage 2)
- [ ] Metrics reported: ΔBP±SD, MD±SD, MAD, MAPD, CP5, CP10, CP15, R
- [ ] Correlation / regression scatter plot
- [ ] Bland–Altman plot (with normality check and 95% CIs on limits of agreement)

## Module 3B — Performance reporting, calibration-based models

- [ ] Models evaluated (baseline model and proposed / final model)
- [ ] Calibration condition and calibration type (one-point / two-point / multi-point)
- [ ] Static test and dynamic test (yes / no)
- [ ] Time since calibration evaluated (and interval in days)
- [ ] Recalibration trigger (if any)
- [ ] Standardized performance table, stratified by condition (Static / Dynamic / Time after calibration)
- [ ] Metrics reported: ΔBP±SD, MD±SD, MAD, MAPD, CP5, CP10, CP15, R
- [ ] Correlation / regression scatter plot (BP change relative to calibration)
- [ ] Bland–Altman plot (with normality check and 95% CIs on limits of agreement)
