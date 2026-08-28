# P300 Speller — Communication Prosthesis

An end-to-end P300 speller implementing the Farwell & Donchin (1988) Row/Column
paradigm for users with profound motor impairment (e.g. Locked-in Syndrome). It
acquires EEG, conditions and epochs the signal, extracts P300 features,
classifies target vs. non-target flashes, and decodes typed characters.

> **Design philosophy: design for failure.** Single-trial P300 detection is
> deep in the noise. Usable accuracy comes from *averaging many flash
> repetitions* per character (`n_sequences`) and decoding the *intersection* of
> the best-scoring row and column. Increase `n_sequences` under fatigue.

## Architecture

```
p300_speller/
├── configs/config.yaml          # every system parameter
├── arduino/
│   └── p300_eeg_acquisition.ino # ADS1115 firmware: timestamped packets + sync markers
├── docs/
│   ├── HARDWARE.md              # BOM, locked circuit parameters, disclosed risks
│   └── VALIDATION.md            # CV policy, artifact criterion, what is unmeasured
├── src/
│   ├── acquisition.py            # Serial reader + synthetic Simulator (shared interface)
│   ├── processing.py             # band-pass, notch, epoching, baseline, artifact rejection
│   ├── features.py               # spatial filter, downsample, flatten
│   ├── classifier.py             # StandardScaler + LDA/LinearSVC, grouped CV, save/load
│   ├── stimulus.py               # pygame 6x6 grid, row/column flashing, perf_counter markers
│   ├── monitor.py                # standalone operator display (causal filter, separate process)
│   ├── session.py                # session manager: phases, prompts, logging
│   └── main_pipeline.py          # train / spell orchestration + event hooks
├── tests/
│   ├── test_processing.py           # filtering + epoching unit tests
│   ├── test_features.py             # spatial filter / downsample / flatten unit tests
│   ├── test_artifact_rejection.py   # peak-to-peak criterion, boundaries, session alarm
│   ├── test_classifier.py           # grouped CV, leakage gate, optimism control
│   ├── test_acquisition_scaling.py  # counts → microvolts incl. the amplifier gain
│   ├── test_monitor.py              # causal vs zero-phase filtering, UDP feed
│   └── test_pipeline_integration.py # rejection is actually wired into the pipeline
├── run.py                        # CLI entry point
└── requirements.txt
```

**Signal path (identical for train and spell):**
`acquire → band-pass + notch → epoch (-100..800 ms) + baseline → reject artifacts
(100 µV p-p) → downsample → flatten → StandardScaler → LDA → average scores per
row/col → argmax intersection`.

**Control layers.** `main_pipeline.P300Pipeline` owns the signal flow and emits
lifecycle events; `session.SessionManager` sits above it to drive operator
prompts, character cycling, and persistent logging (to `output/session_*.log`).

## Hardware

* Arduino Uno + Adafruit ADS1115 16-bit ADC (I2C, 400 kHz).
* Analogue chain: **INA118P** (×23.73) → **TL072** (×40) = **total gain 949**.
* **LM393 photodiode comparator** on D2 for true optical stimulus onset timing.
* 1 channel (Pz) or 3 channels (Fz, Cz, Pz), ground at Fpz.
* Flash the firmware in `arduino/p300_eeg_acquisition.ino` (set `NUM_CHANNELS`
  and `SAMPLE_RATE_HZ` to match `config.yaml`).

The firmware streams `D,<seq>,<micros>,<marker>,<ch0>[,<ch1>,<ch2>]` lines at
115200 baud, plus `S` (optical onset), `T` (timing self-test), `Z` (DC baseline)
and `P` (photodiode diagnostics) lines. Markers use a **host-arms /
photodiode-latches** split: the host supplies stimulus *identity*, the
comparator ISR supplies *timing*.

`acquisition.amplifier_gain` **must** match the built circuit — counts are
divided by it to recover scalp microvolts. See **[docs/HARDWARE.md](docs/HARDWARE.md)**
for the full BOM, the locked circuit parameters, the four disclosed risks, and
the list of figures that are designed rather than measured.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Headless end-to-end smoke test on the synthetic Simulator (no hardware/display)
python run.py selftest

# Calibrate on known words, fitting and saving a model
python p300_speller/run.py train

# Free-text spelling with the trained model
python run.py spell

# Offline simulator demo of free spelling against a known intent
python run.py spell --simulate-intent CAT --chars 3
```

Set `acquisition.use_simulator: false` in `config.yaml` to use the real
Arduino, and `acquisition.serial.port` to your device's port.

### Live operator monitor (optional)

Set `monitor.enabled: true` in `config.yaml`, then in a second terminal:

```bash
python p300_speller/src/monitor.py --config p300_speller/configs/config.yaml
```

It shows per-channel RMS and peak-to-peak against the rejection threshold, plus
the live rejection rate. It runs in its own process, applies its own **causal**
display filter, and never touches the analysis path — no reported result is
attributable to it.

## Tests

```bash
pytest -q            # 88 unit + integration tests
```

The headless `selftest` is the integration check: it calibrates and then spells
on the Simulator, asserting the decoded text matches the intended word.

## Validation and data quality

Performance is estimated with **grouped** cross-validation (`GroupKFold`, one
group per cued character block), pooled into a single ROC. A shuffled estimate
is computed alongside it purely as a **control**, reported as `optimism`, and is
never a result. Grouped AUC above `classifier.leakage_gate_auc` (0.95) is
flagged as a probable leakage defect rather than reported as skill.

Epochs whose peak-to-peak amplitude exceeds **100 µV** on any channel are
rejected inside the shared data path, so the criterion applies to both
calibration and spelling. A session rejection rate above **30%** raises a
`session_invalid` alarm: repeat the recording rather than analysing it.

> An earlier revision used `StratifiedKFold(shuffle=True)` over flat epochs.
> The 94% accuracy figure that produced is a leakage artefact and must not be
> quoted. Realistic three-channel single-trial AUC is **0.65–0.85**.

See **[docs/VALIDATION.md](docs/VALIDATION.md)** for the full policy, the
decimation arithmetic (including the 36 ms tail truncation), and an explicit
list of what is measured versus assumed.

## Key parameters (`config.yaml`)

| Parameter | Meaning |
|---|---|
| `acquisition.sampling_rate_hz` | ADC sample rate (250 Hz); must match firmware |
| `acquisition.amplifier_gain` | total analogue gain (949); divides counts → µV |
| `processing.bandpass` | 0.5–30 Hz Butterworth (zero-phase) |
| `processing.notch.freq_hz` | mains hum (50 EU / 60 NA) |
| `processing.epoch` | window (−100..800 ms) + baseline |
| `processing.artifact_rejection.threshold_uv` | peak-to-peak reject limit (100 µV) |
| `processing.artifact_rejection.max_session_rate` | session-invalidity alarm (0.30) |
| `features.downsample_hz` | epoch decimation (20.83 Hz realised → 54 features) |
| `features.spatial_filter` | `none` (default, 1–3 ch) or `car` (dense montage) |
| `classifier.model_type` | `lda` (default) or `svm` |
| `classifier.cv_strategy` / `cv_folds` | `grouped` (GroupKFold) / fold count |
| `classifier.leakage_gate_auc` | grouped AUC above this is flagged as a bug (0.95) |
| `monitor.enabled` | publish the live operator feed over loopback UDP |
| `speller.n_sequences` | flash repetitions per character (more = slower, more accurate) |
| `speller.flash_duration_ms` / `inter_stimulus_interval_ms` | stimulus timing |

## Safety note

This is research/assistive software. Clinical deployment requires medical-grade,
isolated EEG hardware, formal validation with the individual user, and an
error-correction protocol (a `_`/backspace symbol is included in the matrix).
