"""Component sizing, averaged small-signal models and compensator design.

Small-signal models are the standard state-space-averaged CCM control-to-output
transfer functions (Erickson and Maksimovic, chapters 7 to 9). Compensators use
the k-factor method (Venable) to hit a crossover frequency and phase margin.
"""
from __future__ import annotations
import numpy as np
import control as ct
from .converters import ConverterSpec


# ---------------------------------------------------------------- sizing
def size_inductor(spec: ConverterSpec, di_ratio: float = 0.3) -> float:
    """Inductance so peak-to-peak ripple = di_ratio * average inductor current (CCM)."""
    D, T, Vin = spec.D, spec.T, spec.Vin
    Vo = spec.ideal_vout()
    Io = Vo / spec.R
    if spec.topology == "buck":
        IL, dv = Io, (Vin - Vo) * D * T
    elif spec.topology == "boost":
        IL, dv = Io / (1 - D), Vin * D * T
    else:
        IL, dv = Io / (1 - D), Vin * D * T
    return dv / (di_ratio * IL)


def size_capacitor(spec: ConverterSpec, dv_ratio: float = 0.01, L: float | None = None) -> float:
    """Capacitance so output ripple = dv_ratio * Vout (CCM, ESR neglected)."""
    L = spec.L if L is None else L
    D, T = spec.D, spec.T
    Vo = spec.ideal_vout()
    dv = dv_ratio * Vo
    if spec.topology == "buck":
        di = (spec.Vin - Vo) * D * T / L
        return di * T / (8 * dv)
    return Vo * D * T / (spec.R * dv)


def ccm_boundary_load(spec: ConverterSpec) -> float:
    """Load resistance at the CCM/DCM boundary. R below this value keeps CCM."""
    D, T, L = spec.D, spec.T, spec.L
    if spec.topology == "buck":
        return 2 * L / ((1 - D) * T)
    if spec.topology == "boost":
        return 2 * L / (D * (1 - D) ** 2 * T)
    return 2 * L / ((1 - D) ** 2 * T)


def ccm_boundary_inductance(spec: ConverterSpec) -> float:
    """Critical inductance for CCM at the spec's load."""
    D, T, R = spec.D, spec.T, spec.R
    if spec.topology == "buck":
        return (1 - D) * T * R / 2
    if spec.topology == "boost":
        return D * (1 - D) ** 2 * T * R / 2
    return (1 - D) ** 2 * T * R / 2


# ------------------------------------------------------- small-signal models
def _esr_zero(spec):
    return ct.tf([spec.ESR * spec.C, 1], [1]) if spec.ESR > 0 else ct.tf([1], [1])


def buck_small_signal(spec: ConverterSpec) -> ct.TransferFunction:
    """Control-to-output Gvd(s) = Vin / (1 + s L/R + s^2 L C) times the ESR zero."""
    L, C, R = spec.L, spec.C, spec.R
    return spec.Vin * _esr_zero(spec) / ct.tf([L * C, L / R, 1], [1])


def boost_small_signal(spec: ConverterSpec) -> ct.TransferFunction:
    """Gvd(s) = (Vo/D') (1 - s L/(D'^2 R)) / (1 + s L/(D'^2 R) + s^2 L C/D'^2)."""
    L, C, R, D = spec.L, spec.C, spec.R, spec.D
    Dp = 1 - D
    Vo = spec.Vin / Dp
    num = ct.tf([-L / (Dp ** 2 * R), 1], [1])
    den = ct.tf([L * C / Dp ** 2, L / (Dp ** 2 * R), 1], [1])
    return (Vo / Dp) * num * _esr_zero(spec) / den


def buckboost_small_signal(spec: ConverterSpec) -> ct.TransferFunction:
    L, C, R, D = spec.L, spec.C, spec.R, spec.D
    Dp = 1 - D
    Vo = spec.Vin * D / Dp
    num = ct.tf([-D * L / (Dp ** 2 * R), 1], [1])
    den = ct.tf([L * C / Dp ** 2, L / (Dp ** 2 * R), 1], [1])
    return (Vo / (D * Dp)) * num * _esr_zero(spec) / den


def control_to_output(spec: ConverterSpec) -> ct.TransferFunction:
    return {"buck": buck_small_signal, "boost": boost_small_signal,
            "buckboost": buckboost_small_signal}[spec.topology](spec)


def boost_rhp_zero(spec: ConverterSpec) -> float:
    """Right-half-plane zero of the boost Gvd in rad/s: w_z = D'^2 R / L."""
    return (1 - spec.D) ** 2 * spec.R / spec.L


# ------------------------------------------------------------ compensators
def loop_metrics(loop: ct.TransferFunction) -> dict:
    gm, pm, wcg, wcp = ct.margin(loop)
    return {"gain_margin_dB": 20 * np.log10(gm) if np.isfinite(gm) else np.inf,
            "phase_margin_deg": pm, "crossover_hz": wcp / (2 * np.pi)}


def _phase_at(G, w):
    """Unwrapped phase in degrees at w, continuous from DC (so a two-pole plant reads near -180)."""
    grid = np.logspace(np.log10(w) - 6, np.log10(w), 2000)
    ph = np.degrees(np.unwrap(np.angle(G(1j * grid))))
    return ph[-1]


def _mag_at(G, w):
    return abs(G(1j * w))


def design_pi(plant: ct.TransferFunction, fc: float, zero_ratio: float = 0.2):
    """PI compensator Gc = Kp (1 + wz/s) with zero at zero_ratio*fc and unity loop gain at fc.

    Returns (Gc, metrics). Phase margin is whatever the plant leaves; use it on
    plants that are near first order at fc (e.g. current loops)."""
    wc = 2 * np.pi * fc
    wz = zero_ratio * wc
    Gc0 = ct.tf([1, wz], [1, 0])
    Kp = 1.0 / _mag_at(Gc0 * plant, wc)
    Gc = Kp * Gc0
    return Gc, loop_metrics(Gc * plant)


def design_type2(plant, fc, pm_target=60.0):
    """Type II (one zero, one pole plus integrator) via the k-factor method."""
    wc = 2 * np.pi * fc
    boost = pm_target - 90.0 - _phase_at(plant, wc)  # phase the compensator must add above -90
    boost = np.clip(boost, 0.0, 89.0)
    k = np.tan(np.radians(boost / 2 + 45))
    wz, wp = wc / k, wc * k
    Gc0 = ct.tf([1 / wz, 1], [1]) / (ct.tf([1, 0], [1]) * ct.tf([1 / wp, 1], [1]))
    Kc = 1.0 / _mag_at(Gc0 * plant, wc)
    Gc = Kc * Gc0
    return Gc, loop_metrics(Gc * plant)


def design_type3(plant, fc, pm_target=60.0):
    """Type III (two zeros, two poles plus integrator) via the k-factor method."""
    wc = 2 * np.pi * fc
    boost = pm_target - 90.0 - _phase_at(plant, wc)
    boost = np.clip(boost, 0.0, 179.0)
    k = np.tan(np.radians(boost / 4 + 45)) ** 2
    wz, wp = wc / np.sqrt(k), wc * np.sqrt(k)
    z = ct.tf([1 / wz, 1], [1]); p = ct.tf([1 / wp, 1], [1])
    Gc0 = z * z / (ct.tf([1, 0], [1]) * p * p)
    Kc = 1.0 / _mag_at(Gc0 * plant, wc)
    Gc = Kc * Gc0
    return Gc, loop_metrics(Gc * plant)


def phase_deg(G, w):
    """Unwrapped phase in degrees of transfer function G on rad/s grid w."""
    return np.degrees(np.unwrap(np.angle(G(1j * w))))
