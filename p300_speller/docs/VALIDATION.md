# Validation policy and data-quality gates

This file states exactly how performance is estimated, what data is discarded
before it reaches the classifier, and which numbers in this project are
measured versus assumed. It exists so the methodology and the running code have
a single shared source of truth.

Every claim below is enforced by code and covered by tests in `tests/`.

## 1. Cross-validation: grouped, pooled, with a shuffled control

**The unit of independence is the character block, not the epoch.** The twelve
flashes inside one cued character share electrode drift, impedance, and the
user's attentional state. Splitting them across train and test lets the
classifier recognise the block rather than the neural response.

`classifier.P300Classifier.fit` therefore:

1. Uses **`GroupKFold`**, with one group per cued character block. Groups are
   assigned in `main_pipeline.P300Pipeline.train`.
2. Builds **one pooled ROC** from out-of-fold decision values via
   `cross_val_predict(..., method="decision_function")`, rather than averaging
   per-fold AUCs — folds with few targets cannot then distort the estimate.
3. Also computes a **shuffled `StratifiedKFold`** estimate, retained **only as
   a control**. It is reported as `shuffled_auc` and as
   `optimism = shuffled_auc − grouped_auc`. It is never a result.
4. Uses `n_splits = min(cv_folds, n_groups)`, and returns **NaN with a
   warning** when fewer than two groups exist. A grouped estimate is refused
   rather than faked.
5. Applies a **leakage gate**: grouped AUC above
   `classifier.leakage_gate_auc` (0.95) sets `leakage_flag` and emits a
   `RuntimeWarning`. On a three-channel montage that is a bug to investigate,
   not skill to report.

`TrainingReport` carries `grouped_auc`, `shuffled_auc`, `optimism`, `n_groups`,
`grouped_folds`, and `leakage_flag`.

### The discarded 94% figure

The previous implementation used `StratifiedKFold(shuffle=True)` over flat
epochs. That is the leakage mechanism, and the 94% accuracy figure it produced
**is not a result and must not be quoted anywhere**.

`tests/test_classifier.py::test_shuffled_control_is_optimistic_when_effect_is_block_specific`
demonstrates the mechanism directly on synthetic data: when most of the
discriminative signal is block-specific, the grouped estimate lands near chance
(0.550) while the shuffled estimate reads 0.681 — an optimism of +0.131 from
nothing but the choice of splitter.

### Honest bounds

| Condition | Expected AUC |
|---|---|
| 3 channels, low-cost hardware | **0.65 – 0.85** |
| Above 0.95 | Leakage, not skill — investigate |

## 2. Artifact rejection: 100 µV peak-to-peak

`processing.reject_artifacts` discards an epoch when the peak-to-peak amplitude
of **any** channel exceeds the threshold. Configured in
`processing.artifact_rejection`:

| Key | Value |
|---|---|
| `threshold_uv` | **100.0** |
| `max_session_rate` | **0.30** |

**Peak-to-peak, not absolute deviation.** The distinction is not cosmetic. A
slow 90 µV drift ramp has a maximum absolute deviation of only 45 µV once the
epoch is baseline-corrected, so a ±75 µV absolute bound keeps it — and drift is
exactly the contamination that must be removed. `tests/test_artifact_rejection.py`
pins this with the arithmetic worked through explicitly.

**Boundary.** Reject if it *exceeds* the threshold; an epoch sitting exactly at
100 µV is kept.

**Session alarm.** A rejection rate **above** 30% means the session is invalid
and must be repeated, not analysed. The rate is accumulated across the whole
session (`P300Pipeline.session_rejection_rate`), not per character, and
`_check_session_quality` fires a `session_invalid` event before any model is fit
or saved.

**It is wired in, not merely defined.** Rejection runs inside
`P300Pipeline._collect_epochs`, the single path shared by calibration and free
spelling, so a contaminated epoch cannot reach the classifier by either route.
`tests/test_pipeline_integration.py` asserts this against the real pipeline
rather than the function in isolation.

## 3. Decimation arithmetic and the 36 ms tail

Decimation is by an **integer** factor, so the requested rate and the realised
rate differ:

```
factor            = round(250 / 20.83)   = 12
effective rate    = 250 / 12             = 20.83 Hz     (not 20.0 Hz)
samples per epoch = round(0.9 × 250)     = 225
usable samples    = (225 // 12) × 12     = 216
retained points   = 216 / 12             = 18
feature vector    = 18 × 3 channels      = 54 elements
```

**The epoch is configured to +800 ms; the feature vector spans only to
+764 ms.** Block averaging consumes whole blocks, so the final 9 samples —
**36 ms** — are discarded from the end of every epoch. This is harmless for the
P300 itself (peak near +300 ms) but it is not invisible: anyone multiplying 18
points by 48 ms will find it. State it wherever the epoch window is described.

The code was always correct here; only the documentation was wrong. Pinned by
`tests/test_features.py::test_decimation_discards_36ms_tail`.

## 4. Filter order

As implemented in `processing.precondition`: **band-pass, then notch.** Both
stages are zero-phase, so the order affects neither phase nor latency.

The 50 Hz notch downstream of a 30 Hz low-pass is largely redundant *in the
analysis path*. It is retained because it costs nothing, it guards against the
band being widened in `config.yaml` without revisiting the function, and it is
genuinely necessary in the display path, where `monitor.py` uses a causal
filter with far less stop-band attenuation.

> **Open item.** Confirm the order asserted in the methodology text matches the
> statement above. This has not been verified against the written chapters.

## 5. Zero-phase analysis vs causal display

Two filters exist deliberately:

| | Analysis (`processing.py`) | Display (`monitor.py`) |
|---|---|---|
| Function | `sosfiltfilt` / `filtfilt` | `sosfilt` with carried `zi` |
| Causality | Non-causal (forward + backward) | Causal |
| Group delay | **Identically zero** at every frequency | Non-zero, frequency-dependent |
| Used for results | Yes | **Never** |

Forward–backward filtering conjugates the phase response, so the net phase is
zero and latency is preserved — which is why it is used for analysis. A causal
filter shifts and disperses the waveform, which would bias any latency
measurement, but it is the correct choice for a display that must show the
operator what is happening *now* and cannot wait for future samples.

`tests/test_monitor.py::test_causal_display_filter_delays_but_zero_phase_does_not`
demonstrates the difference on an impulse.

**No reported result is attributable to the display filter.** The monitor runs
in a separate process, receives a copy of the raw stream over loopback UDP, and
never writes to the model, the epoch store, or the result files.

## 6. Operator monitor

Enabled via `monitor.enabled: true` in `config.yaml`. The pipeline publishes
fire-and-forget UDP datagrams; `src/monitor.py` listens and renders.

```bash
# Terminal 1
python p300_speller/run.py train
# Terminal 2
python p300_speller/src/monitor.py --config p300_speller/configs/config.yaml
```

The split exists because only one process may hold the serial port, and drawing
must not compete with the pygame stimulus loop or the serial reader thread. If
the monitor is absent, slow, or dies, the recording is unaffected.

## 7. Reference results (external to this repository)

These figures come from an offline **BNCI2014-008** analysis (8 ALS patients,
6×6 row/column paradigm) run outside this codebase. They are recorded here for
traceability. **They are not produced by any script in this repository**, and
nothing here reproduces them.

| Metric | Value |
|---|---|
| 3-channel AUC | 0.710 ± 0.064 |
| 8-channel AUC | 0.867 ± 0.044 |
| Paired *t*-test | t(7) = 11.85, p = 6.9 × 10⁻⁶ |
| Cohen's *d* | 4.19 |
| Character accuracy | 34.4 – 98.5%, mean 69.3% |
| ITR | 7.25 bit/min at 7 sequences |
| Discriminative capacity retention | η = 0.572 |
| Cross-subject LOSO AUC | 0.580 |

Two caveats travel with these numbers:

* **Topography is Fz-dominant** (Fz ≈ 2.0 µV, Cz ≈ 1.2 µV, Pz ≈ 0.1 µV). This
  **contradicts** the Pz-dominant gradient assumed in earlier work — including
  the simulator's channel-gain profile in `acquisition.Simulator`, which still
  encodes Pz > Cz > Fz. The simulator profile is a synthetic convenience, not a
  finding; any figure built on the old Pz-dominant assumption must be
  regenerated or removed.
* **Character accuracy is an upper bound.** It is derived from a 6-alternative
  forced-choice analytical model because MOABB does not expose row/column
  identity for true intersection decoding. Label it as a bound.

## 8. Reproducing what this repository *does* produce

```bash
pip install -r requirements.txt
pytest -q                 # 88 tests
python run.py selftest    # headless end-to-end on the Simulator
```

The `selftest` calibrates on "CAT" and then spells it back, asserting the
decoded text matches. It exercises grouped cross-validation, artifact rejection,
and the full acquire → condition → epoch → reject → vectorise → classify path
on synthetic data. It validates the *plumbing*; it is not evidence about human
EEG.
