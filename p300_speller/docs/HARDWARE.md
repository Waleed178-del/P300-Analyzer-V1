# Hardware — bill of materials, locked parameters, and disclosed risks

This is the authoritative hardware record for the P300 speller front-end. Where
a value here disagrees with any other document, this file and
`configs/config.yaml` win, and the other document is out of date.

## 1. Bill of materials

| Function | Part | Package | Notes |
|---|---|---|---|
| Instrumentation amplifier | **INA118P** | PDIP-8 | Stage 1, one per channel |
| Operational amplifier | **TL072** | dual JFET | Stage 2 + filtering |
| Photodetector / optical sync | **LM393 photodiode comparator module** | module | Stimulus onset timing |
| ADC | **ADS1115** | breakout | 16-bit, I2C, GAIN_ONE (±4.096 V) |
| MCU | Arduino Uno (ATmega328P) | — | Firmware in `arduino/` |

### Superseded parts — do not use

Earlier documentation specified **INA128**, **OPA2350**, and **BPW34**. All
three are superseded by the table above. Any document, schematic, or BOM still
naming them is stale and must be corrected.

## 2. Locked circuit parameters

| Parameter | Value |
|---|---|
| INA118P gain (stage 1) | 23.73 (RG = 2.2 kΩ) |
| TL072 gain (stage 2) | 40 |
| **Total system gain** | **949** |
| High-pass corner | 0.48 Hz |
| Low-pass | Sallen–Key, 33.6 Hz, Q = 0.72 |
| ADC | ADS1115, 16-bit, 250 Hz requested |
| Supply | Single +5 V |
| Input coupling | DC coupled |
| Ground electrode | Fpz |
| Impedance target | 10 kΩ |
| Board | Single-sided PCB, no ground plane |

The gain of 949 is consumed by software as `acquisition.amplifier_gain` in
`configs/config.yaml` and is divided out in
`acquisition.SerialAcquisition._counts_to_uv`. Changing the circuit **requires**
changing that key, or every reported amplitude becomes wrong by the ratio.

### Counts → microvolts

```
volts per count  = 4.096 V / 32768          = 125 µV   (at the ADC input)
scalp microvolts = counts × 125 µV / 949    ≈ 0.1317 µV per count
```

A 5 µV scalp P300 reaches the ADC as ≈ 4.7 mV. Omitting the ÷949 reports it as
≈ 4745 µV, which is also why the omission silently disables the 100 µV
peak-to-peak artifact criterion — nothing real ever crosses it.

## 3. Disclosed risks

These are real limitations of the build and belong in the methodology, not in a
footnote.

1. **Limited stage-1 headroom for electrode offset.** A stage-1 gain of 23.73
   on a single +5 V supply leaves little room for electrode DC offset drift. Up
   to ~50 mV of offset saturates the INA118P before the 0.48 Hz high-pass can
   act on it. Characterise the offset before every session with the firmware's
   `Z` command (DC baseline routine).

2. **Thin mains rejection.** A two-pole 33.6 Hz Sallen–Key provides only roughly
   12–15 dB of attenuation at 50 Hz at full system gain. That is thin, and it is
   why the digital notch is retained downstream.

3. **Inter-channel skew.** The ADS1115 has one converter behind an input
   multiplexer, so Fz, Cz and Pz are **not** sampled simultaneously. At 860 SPS
   each conversion costs ≈ 1.16 ms, so Pz lags Fz by ≈ 2.3 ms per frame. This
   systematically smears measured P300 latency across channels and is **not**
   corrected in firmware or software. The firmware `T` self-test reports the
   measured per-channel conversion time so the true skew can be stated.

4. **INA118P REF pin requires buffered drive.** The REF pin sits on an internal
   25 kΩ network; driving it from an unbuffered VMID divider destroys CMRR.
   This is a correctness requirement, not a design preference.

5. **Single-sided PCB, no ground plane.** Omitting a ground plane measurably
   worsens 50 Hz common-mode rejection in a microvolt-level circuit. This was
   accepted for local fabrication reasons. State it explicitly; do not bury it.

## 4. What has NOT been measured

Four figures in this document are **designed or calculated, never measured**.
No result that depends on them may be presented as instrument validation.

| Quantity | Status | How to close it |
|---|---|---|
| Total gain = 949 | Designed from component values | Bench measurement, per channel, with a known input |
| Sample rate = 250 Hz | Compile-time `#define`, a request | Firmware `T` command reports the **achieved** rate |
| Mains attenuation 12–15 dB | Calculated from the filter response | Swept measurement at the electrode input |
| Inter-channel skew | Derived from the datasheet conversion time | Firmware `T` command reports measured conversion time |

Electrode procurement has not happened, so no bench characterisation has begun.
This is the single largest open gap in the work and belongs at the top of the
future-work section.

## 5. Firmware interface

Flash `arduino/p300_eeg_acquisition.ino`. Set `NUM_CHANNELS` and
`SAMPLE_RATE_HZ` to match `config.yaml`.

### Device → host

| Line | Meaning |
|---|---|
| `D,<seq>,<micros>,<marker>,<ch0>[,<ch1>,<ch2>]` | Data sample. **Format frozen** — the Python decoder depends on it. |
| `S,<seq>,<onset_us>,<code>,<arm_latency_us>` | True optical stimulus onset from the photodiode ISR |
| `H,P300_EEG,<rate>,<n_ch>,<fw>` | Boot header |
| `I,<key>=<value>,...` | Configuration readout |
| `M,<code>,<micros>` | Marker-armed acknowledgement |
| `T,<key>=<value>,...` | Timing self-test result |
| `Z,<ch>,<mean_counts>,<offset_uv>` | DC baseline / electrode offset |
| `P,<key>=<value>,...` | Photodiode diagnostics |
| `E,<reason>` | Error |

### Host → device

| Command | Effect |
|---|---|
| `M<code>` | Arm marker `<code>` for the next optical onset |
| `T` | Run the timing self-test, report the achieved rate |
| `Z` | Run the DC baseline routine |
| `P` | Report photodiode diagnostics |
| `I` | Report configuration |

### Marker timing: host arms, photodiode latches

Both halves are required and neither is sufficient:

* The **host** supplies stimulus **identity** — only it knows which row or
  column is about to flash, so it arms a code before the flash.
* The **photodiode** supplies **timing** — the LM393 fires INT0 when the screen
  actually goes bright, and the ISR timestamps that edge. The comparator cannot
  know *what* flashed, only *when*.

The previous firmware latched the host's marker onto the next emitted sample.
That path carries USB serial transit, host scheduler jitter, and up to one full
sample period of quantisation — 4 ms at 250 Hz from quantisation alone, before
transit. Timestamping the optical edge in an ISR removes all of it.

For backward compatibility the armed code is still latched onto the next `D`
packet, so a host that ignores `S` lines behaves exactly as before.

## 6. Electrical safety

Research-grade hardware, **not certified to IEC 60601**. State this in both the
consent form and the thesis.

* Laptop on **battery**, charger unplugged, no docking station.
* USB isolator of **ADuM3160 class, rated 2.5 kV or better**, in line.
* No other mains-powered device touching the subject or the leads.
* Operator present for the entire session.
