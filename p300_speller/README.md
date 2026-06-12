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
├── src/
│   ├── eeg_acquisition.py        # Serial reader + synthetic Simulator (shared interface)
│   ├── signal_processing.py      # Butterworth band-pass, notch, epoching, baseline
│   ├── feature_extraction.py     # spatial filter, downsample, flatten
│   ├── classifier.py             # StandardScaler + LDA/LinearSVC, save/load, decoding
│   ├── speller_matrix.py         # pygame 6x6 grid, row/column flashing, perf_counter markers
│   └── main_pipeline.py          # train / spell orchestration
└── run.py                        # CLI entry point
```

**Signal path (identical for train and spell):**
`acquire → band-pass + notch → epoch (-100..800 ms) + baseline → downsample →
flatten → StandardScaler → LDA → average scores per row/col → argmax intersection`.

## Hardware

* Arduino Uno + Adafruit ADS1115 16-bit ADC (I2C).
* 1 channel (Pz) or 3 channels (Fz, Cz, Pz) referenced to ear/mastoid + ground.
* Flash the firmware in `arduino/p300_eeg_acquisition.ino` (set `NUM_CHANNELS`
  and `SAMPLE_RATE_HZ` to match `config.yaml`).

The firmware streams `D,<seq>,<micros>,<marker>,<ch0>[,<ch1>,<ch2>]` lines at
115200 baud and accepts `M<marker>\n` sync commands from the host so each flash
is hardware-aligned to a real sample.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Headless end-to-end smoke test on the synthetic Simulator (no hardware/display)
python run.py selftest

# Calibrate on known words, fitting and saving a model
python run.py train

# Free-text spelling with the trained model
python run.py spell

# Offline simulator demo of free spelling against a known intent
python run.py spell --simulate-intent CAT --chars 3
```

Set `acquisition.use_simulator: false` in `config.yaml` to use the real
Arduino, and `acquisition.serial.port` to your device's port.

## Key parameters (`config.yaml`)

| Parameter | Meaning |
|---|---|
| `acquisition.sampling_rate_hz` | ADC sample rate (250 Hz); must match firmware |
| `processing.bandpass` | 0.5–30 Hz Butterworth (zero-phase) |
| `processing.notch.freq_hz` | mains hum (50 EU / 60 NA) |
| `processing.epoch` | window (−100..800 ms) + baseline |
| `features.downsample_hz` | epoch decimation (20 Hz) |
| `classifier.model_type` | `lda` (default) or `svm` |
| `speller.n_sequences` | flash repetitions per character (more = slower, more accurate) |
| `speller.flash_duration_ms` / `inter_stimulus_interval_ms` | stimulus timing |

## Safety note

This is research/assistive software. Clinical deployment requires medical-grade,
isolated EEG hardware, formal validation with the individual user, and an
error-correction protocol (a `_`/backspace symbol is included in the matrix).
