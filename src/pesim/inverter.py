"""Three-phase two-level voltage source inverter: SPWM and SVPWM waveforms and spectra.

Pole voltages are referenced to the DC-link midpoint and take values +Vdc/2 or
-Vdc/2. Line-to-line voltage is the difference of two pole voltages.

Dead time note: real gate drivers insert a blanking interval (typically 0.5 to
3 us) between turning one switch off and its complement on. During dead time the
output is clamped by the freewheeling diode, so the pole voltage depends on the
current sign. The net effect is a voltage error of magnitude Vdc*td*fsw per
period whose sign follows the current, adding low-order odd harmonics (5th,
7th) and reducing the fundamental. This module models ideal switching only; the
helper dead_time_voltage_error() returns the average error magnitude.
"""
from __future__ import annotations
import numpy as np


def _carrier(t, fsw):
    """Triangular carrier in [-1, 1]."""
    return 2 * np.abs(2 * ((t * fsw) % 1.0) - 1) - 1


def spwm(Vdc, m, f1=50.0, fsw=5e3, n_periods=1, fs=None, t=None):
    """Sinusoidal PWM. Returns dict with t, va, vb, vc (pole voltages), refs."""
    if t is None:
        fs = fs or 200 * fsw
        t = np.arange(0, n_periods / f1, 1 / fs)
    ph = 2 * np.pi * f1 * t
    refs = [m * np.sin(ph - k * 2 * np.pi / 3) for k in range(3)]
    car = _carrier(t, fsw)
    poles = [np.where(r > car, Vdc / 2, -Vdc / 2) for r in refs]
    return {"t": t, "va": poles[0], "vb": poles[1], "vc": poles[2],
            "refs": refs, "carrier": car, "f1": f1}


def svpwm(Vdc, m, f1=50.0, fsw=5e3, n_periods=1, fs=None, t=None):
    """Space vector PWM implemented as SPWM with min-max zero-sequence injection.

    m is defined on the same scale as SPWM (m = 1 gives fundamental Vdc/2 phase
    peak). The linear range extends to m = 2/sqrt(3) = 1.1547."""
    if t is None:
        fs = fs or 200 * fsw
        t = np.arange(0, n_periods / f1, 1 / fs)
    ph = 2 * np.pi * f1 * t
    sines = np.array([np.sin(ph - k * 2 * np.pi / 3) for k in range(3)])
    zs = -0.5 * (sines.max(axis=0) + sines.min(axis=0))
    refs = [m * (s + zs) for s in sines]
    car = _carrier(t, fsw)
    poles = [np.where(r > car, Vdc / 2, -Vdc / 2) for r in refs]
    return {"t": t, "va": poles[0], "vb": poles[1], "vc": poles[2],
            "refs": refs, "carrier": car, "f1": f1}


def line_line(w):
    return w["va"] - w["vb"]


def spectrum(x, t, f1, n_harm=100):
    """Harmonic amplitudes (peak) at k*f1, k = 0..n_harm, via FFT of an integer number of periods."""
    N = len(x)
    dt = t[1] - t[0]
    X = np.fft.rfft(x) / N * 2
    X[0] /= 2
    freqs = np.fft.rfftfreq(N, dt)
    idx = [int(round(k * f1 / (freqs[1]))) for k in range(n_harm + 1)]
    idx = [i for i in idx if i < len(X)]
    return freqs[idx], np.abs(X[idx])


def fundamental(x, t, f1):
    """Peak amplitude of the f1 component by projection."""
    ph = 2 * np.pi * f1 * t
    a = 2 * np.mean(x * np.cos(ph)); b = 2 * np.mean(x * np.sin(ph))
    return float(np.hypot(a, b))


def thd(x, t, f1):
    """Total harmonic distortion including all non-fundamental, non-DC content."""
    x = x - x.mean()
    V1 = fundamental(x, t, f1)
    Vrms = np.sqrt(np.mean(x ** 2))
    return float(np.sqrt(max(Vrms ** 2 - V1 ** 2 / 2, 0.0)) / (V1 / np.sqrt(2)))


def dc_utilisation(Vdc, m, scheme="spwm"):
    """Peak fundamental line-to-line voltage divided by Vdc (linear range only)."""
    m_max = 1.0 if scheme == "spwm" else 2 / np.sqrt(3)
    m = min(m, m_max)
    return np.sqrt(3) * m / 2


def dead_time_voltage_error(Vdc, td, fsw):
    """Average pole-voltage error magnitude caused by dead time td at switching frequency fsw."""
    return Vdc * td * fsw
