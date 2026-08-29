"""Tests for the counts-to-microvolts conversion in :mod:`acquisition`.

The conversion must divide by BOTH the ADC scaling and the analogue gain of the
head-stage. Omitting the second division inflates every reported amplitude by
the full gain (949x in the reference build) and, as a direct consequence, makes
the 100 uV peak-to-peak artifact criterion inoperative: a genuine 5 uV scalp
P300 would be reported as ~4745 uV and every epoch would be rejected, while a
threshold scaled to match would reject nothing real.
"""

from __future__ import annotations

import numpy as np
import pytest

import acquisition
from acquisition import SerialAcquisition, build_source

pytestmark = pytest.mark.skipif(
    acquisition.serial is None, reason="pyserial not installed"
)

ADC_FULLSCALE_V = 4.096
ADC_BITS = 16
GAIN = 949.0


def _source(gain: float = GAIN) -> SerialAcquisition:
    return SerialAcquisition(
        port="/dev/null",
        baud_rate=115200,
        sampling_rate_hz=250.0,
        channel_names=["Fz", "Cz", "Pz"],
        ring_buffer_s=1.0,
        adc_fullscale_v=ADC_FULLSCALE_V,
        adc_resolution_bits=ADC_BITS,
        amplifier_gain=gain,
    )


def test_lsb_at_adc_input_is_125_microvolts() -> None:
    """Sanity anchor: 4.096 V over 32768 counts is 125 uV per LSB AT THE ADC."""
    volts_per_count = ADC_FULLSCALE_V / float(2 ** (ADC_BITS - 1))
    assert volts_per_count * 1e6 == pytest.approx(125.0)


def test_scale_factor_divides_out_the_amplifier_gain() -> None:
    src = _source()
    expected = (ADC_FULLSCALE_V / float(2 ** (ADC_BITS - 1))) * 1e6 / GAIN
    assert src._counts_to_uv == pytest.approx(expected)
    # ~0.132 uV per count referred to the scalp, not 125 uV.
    assert src._counts_to_uv == pytest.approx(0.13172, rel=1e-3)


def test_scalp_referred_amplitude_is_physiological() -> None:
    """A 5 uV scalp P300 must decode back to ~5 uV, not ~4745 uV."""
    src = _source()
    counts = (5.0e-6 * GAIN) / (ADC_FULLSCALE_V / float(2 ** (ADC_BITS - 1)))
    assert counts * src._counts_to_uv == pytest.approx(5.0, rel=1e-9)


def test_omitting_the_gain_inflates_by_949x() -> None:
    """Pinning the size of the defect this divisor fixes."""
    with_gain = _source(GAIN)._counts_to_uv
    without_gain = _source(1.0)._counts_to_uv
    assert without_gain / with_gain == pytest.approx(949.0)


def test_decode_applies_the_scalp_referred_scale() -> None:
    src = _source()
    line = b"D,1,1000,0,100,-200,300\n"
    out = src._decode(line)
    np.testing.assert_allclose(
        out, np.array([100.0, -200.0, 300.0]) * src._counts_to_uv
    )


def test_decode_raises_on_payload_mismatch() -> None:
    """A truncated hardware payload is a fault, not something to skip past."""
    src = _source()
    with pytest.raises(ValueError, match="Hardware payload mismatch"):
        src._decode(b"D,1,1000,0,100\n")          # 3 channels expected, 1 given


def test_decode_ignores_non_data_lines() -> None:
    """Header/info/status lines from the firmware are not payload faults."""
    src = _source()
    for line in (b"H,version=2\n", b"I,gain=949\n", b"T,rate=217.4\n", b"\n"):
        assert src._decode(line) is None


def test_non_positive_gain_is_rejected() -> None:
    with pytest.raises(ValueError, match="amplifier_gain"):
        _source(0.0)


def test_build_source_passes_gain_from_config() -> None:
    config = {
        "acquisition": {
            "channels": ["Fz", "Cz", "Pz"],
            "sampling_rate_hz": 250,
            "ring_buffer_s": 1.0,
            "use_simulator": False,
            "adc_fullscale_v": ADC_FULLSCALE_V,
            "adc_resolution_bits": ADC_BITS,
            "amplifier_gain": 500.0,
            "serial": {"port": "/dev/null", "baud_rate": 115200,
                       "read_timeout_s": 1.0},
        }
    }
    src = build_source(config)
    assert isinstance(src, SerialAcquisition)
    assert src.amplifier_gain == 500.0
