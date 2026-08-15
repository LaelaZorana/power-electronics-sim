"""Switched state-space time-domain simulation of buck, boost and buck-boost converters.

The state vector is x = [iL, vC]. Each converter has three affine sub-circuits:
switch on, switch off (diode conducting) and idle (inductor current zero, DCM).
Each sub-circuit is integrated with its exact zero-order-hold discretisation
computed from a matrix exponential, so the time step only affects how finely
the switching instants and the DCM boundary are resolved.

Loss elements: MOSFET on resistance Rds_on, inductor winding resistance R_L,
diode forward drop V_f, output capacitor ESR. Set them to zero for the ideal case.

Conventions and resolution notes. The buck-boost output is reported as a
positive magnitude, the physical output of the inverting topology is negative.
The requested duty cycle is realised as an integer number of on steps, the
effective value is stored on SimResult.D_eff and a warning is issued when it
differs from the request by more than 1e-3. The DCM boundary is resolved to one
time step, the clamp zeroes iL at the step boundary without correcting vC for
the partial step.
"""
from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from typing import Literal
import numpy as np
from scipy.linalg import expm

Converter = Literal["buck", "boost", "buckboost"]


@dataclass
class ConverterSpec:
    topology: Converter = "buck"
    Vin: float = 12.0
    D: float = 0.5
    fs: float = 100e3
    L: float = 100e-6
    C: float = 100e-6
    R: float = 10.0
    Rds_on: float = 0.0
    R_L: float = 0.0
    V_f: float = 0.0
    ESR: float = 0.0

    def __post_init__(self):
        if self.topology not in ("buck", "boost", "buckboost"):
            raise ValueError(f"unknown topology {self.topology!r}")
        if not 0.0 <= self.D < 1.0:
            raise ValueError(f"duty cycle D = {self.D} must satisfy 0 <= D < 1")
        if self.Vin <= 0:
            raise ValueError(f"Vin = {self.Vin} must be positive")
        for name in ("fs", "L", "C", "R"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} = {getattr(self, name)} must be positive")
        for name in ("Rds_on", "R_L", "V_f", "ESR"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} = {getattr(self, name)} must not be negative")

    @property
    def T(self):
        return 1.0 / self.fs

    def ideal_vout(self):
        """Ideal CCM output voltage magnitude. The buck-boost value is the
        magnitude of a physically negative output."""
        D = self.D
        return {"buck": D * self.Vin,
                "boost": self.Vin / (1 - D),
                "buckboost": self.Vin * D / (1 - D)}[self.topology]


@dataclass
class SimResult:
    t: np.ndarray
    iL: np.ndarray
    vC: np.ndarray
    vout: np.ndarray
    iin: np.ndarray
    sw: np.ndarray
    spec: ConverterSpec
    steps_per_period: int
    mode: str = ""
    D_eff: float = 0.0
    Vout_avg: float = 0.0
    dv_pp: float = 0.0
    di_pp: float = 0.0
    IL_avg: float = 0.0
    Pin: float = 0.0
    Pout: float = 0.0
    efficiency: float = 0.0
    losses: dict = field(default_factory=dict)

    def last_periods(self, n=1):
        """Slice covering the last n switching periods, clamped to the run length."""
        k = min(n * self.steps_per_period, len(self.t))
        return slice(len(self.t) - k, len(self.t))


def _output_gain(spec):
    return spec.R / (spec.R + spec.ESR)


def _modes(spec: ConverterSpec):
    """Return a dict mapping each mode name to its (A, b, cout) with vout = cout @ x."""
    L, C, R, rc = spec.L, spec.C, spec.R, spec.ESR
    g = _output_gain(spec)
    Rs = spec.Rds_on + spec.R_L
    Vin, Vf = spec.Vin, spec.V_f
    top = spec.topology
    # vout expressed as cout @ [iL, vC], with iC = iL*k - vout/R where k=1 when
    # the inductor feeds the output node, else 0.
    c_fed = np.array([g * rc, g])       # inductor feeds output node
    c_idle = np.array([0.0, g])         # inductor disconnected from output
    modes = {}
    if top == "buck":
        # on: L diL = Vin - iL*Rs - vout ; C dvC = iL - vout/R
        A_on = np.array([[-(Rs + c_fed[0]) / L, -c_fed[1] / L],
                         [(1 - c_fed[0] / R) / C, -c_fed[1] / (R * C)]])
        modes["on"] = (A_on, np.array([Vin / L, 0.0]), c_fed)
        A_off = np.array([[-(spec.R_L + c_fed[0]) / L, -c_fed[1] / L],
                          [(1 - c_fed[0] / R) / C, -c_fed[1] / (R * C)]])
        modes["off"] = (A_off, np.array([-Vf / L, 0.0]), c_fed)
    elif top in ("boost", "buckboost"):
        # on: inductor charged from Vin, output isolated
        A_on = np.array([[-Rs / L, 0.0], [0.0, -c_idle[1] / (R * C)]])
        modes["on"] = (A_on, np.array([Vin / L, 0.0]), c_idle)
        Voff = (Vin - Vf) if top == "boost" else -Vf
        A_off = np.array([[-(spec.R_L + c_fed[0]) / L, -c_fed[1] / L],
                          [(1 - c_fed[0] / R) / C, -c_fed[1] / (R * C)]])
        modes["off"] = (A_off, np.array([Voff / L, 0.0]), c_fed)
    else:
        raise ValueError(top)
    A_idle = np.array([[0.0, 0.0], [0.0, -c_idle[1] / (R * C)]])
    modes["idle"] = (A_idle, np.zeros(2), c_idle)
    return modes


def _discretise(A, b, dt):
    M = np.zeros((3, 3))
    M[:2, :2] = A * dt
    M[:2, 2] = b * dt
    E = expm(M)
    return E[:2, :2], E[:2, 2]


def _period_map(spec, disc, x, steps_per_period, n_on):
    """Advance state x through one switching period."""
    x = np.array(x, float)
    for k in range(steps_per_period):
        if k < n_on:
            mode = "on"
        else:
            mode = "off" if x[0] > 0 else "idle"
        Ad, Bd = disc[mode]
        x = Ad @ x + Bd
        if mode == "off" and x[0] < 0:
            x[0] = 0.0
    return x


def steady_state(spec: ConverterSpec, steps_per_period: int = 200, iters: int = 30):
    """Periodic steady state at the start of a switching period by Newton shooting.

    Solves x = P(x) where P is the one-period map. In CCM the map is affine so
    Newton converges in one step; in DCM a few iterations are needed. Warns
    when the residual of the period map is still above 1e-6 after iters."""
    modes = _modes(spec)
    dt = spec.T / steps_per_period
    disc = {k: _discretise(A, b, dt) for k, (A, b, _) in modes.items()}
    n_on = int(round(spec.D * steps_per_period))
    Vo = spec.ideal_vout()
    x = np.array([max(Vo / spec.R, 0.0), Vo])
    F = np.zeros(2)
    for _ in range(iters):
        Px = _period_map(spec, disc, x, steps_per_period, n_on)
        F = Px - x
        if np.linalg.norm(F) < 1e-10:
            break
        J = np.zeros((2, 2))
        for i in range(2):
            e = np.zeros(2); e[i] = 1e-4 * max(1.0, abs(x[i]))
            J[:, i] = (_period_map(spec, disc, x + e, steps_per_period, n_on) - Px) / e[i]
        try:
            dx = np.linalg.solve(J - np.eye(2), -F)
        except np.linalg.LinAlgError:
            dx = F
        x = x + dx
        x[0] = max(x[0], 0.0)
    if np.linalg.norm(F) > 1e-6:
        warnings.warn(f"steady_state: shooting residual {np.linalg.norm(F):.2e} "
                      f"after {iters} iterations", stacklevel=2)
    return x


def simulate(spec: ConverterSpec, n_periods: int = 20, steps_per_period: int = 200,
             x0=None, from_steady_state: bool = True) -> SimResult:
    """Run a switched simulation and return waveforms plus steady-state metrics.

    By default the initial state is the periodic steady state found by
    steady_state(), so a handful of periods is enough. Pass x0 (or
    from_steady_state=False, which starts from zero) to see the start-up
    transient; then n_periods must cover several R*C time constants.
    Metrics are computed on the last 5 switching periods, or fewer when
    n_periods is below 5. The duty cycle is realised as an integer number of
    on steps, D_eff on the result records the value actually run and a warning
    is issued when it differs from the request by more than 1e-3. The stepping
    is a plain Python loop, exact per step but not vectorised.
    """
    if n_periods < 1:
        raise ValueError("n_periods must be at least 1")
    if steps_per_period < 10:
        raise ValueError("steps_per_period must be at least 10")
    if x0 is None and from_steady_state:
        x0 = steady_state(spec, steps_per_period)
    modes = _modes(spec)
    dt = spec.T / steps_per_period
    disc = {k: _discretise(A, b, dt) for k, (A, b, _) in modes.items()}
    cout = {k: c for k, (_, _, c) in modes.items()}
    n_on = int(round(spec.D * steps_per_period))
    D_eff = n_on / steps_per_period
    if abs(D_eff - spec.D) > 1e-3:
        warnings.warn(f"duty cycle quantised from {spec.D} to {D_eff:.4f} by "
                      f"steps_per_period={steps_per_period}, raise steps_per_period "
                      f"for a finer grid", stacklevel=2)
    N = n_periods * steps_per_period
    x = np.zeros(2) if x0 is None else np.array(x0, float)
    iL = np.empty(N); vC = np.empty(N); vout = np.empty(N)
    iin = np.empty(N); sw = np.empty(N, dtype=np.int8)
    top = spec.topology
    for n in range(N):
        k = n % steps_per_period
        if k < n_on:
            mode = "on"
        else:
            mode = "off" if x[0] > 0 else "idle"
        Ad, Bd = disc[mode]
        iL[n], vC[n] = x
        vout[n] = cout[mode] @ x
        iin[n] = x[0] if (top == "boost" or mode == "on") else 0.0
        sw[n] = {"on": 1, "off": 0, "idle": -1}[mode]
        x = Ad @ x + Bd
        if mode == "off" and x[0] < 0:
            x[0] = 0.0
    t = np.arange(N) * dt
    res = SimResult(t, iL, vC, vout, iin, sw, spec, steps_per_period)
    res.D_eff = D_eff
    s = res.last_periods(5)
    res.Vout_avg = float(vout[s].mean())
    res.dv_pp = float(vout[s].max() - vout[s].min())
    res.di_pp = float(iL[s].max() - iL[s].min())
    res.IL_avg = float(iL[s].mean())
    res.mode = "DCM" if (sw[s] == -1).any() else "CCM"
    res.Pout = float((vout[s] ** 2 / spec.R).mean())
    res.Pin = float((spec.Vin * iin[s]).mean())
    res.efficiency = res.Pout / res.Pin if res.Pin > 0 else float("nan")
    on = sw[s] == 1; off = sw[s] == 0
    if top == "buck":
        ic = iL[s] - vout[s] / spec.R          # inductor always feeds the output
    else:
        ic = iL[s] * (~on) - vout[s] / spec.R  # inductor feeds the output only off/idle
    res.losses = {
        "mosfet_conduction": float((iL[s] ** 2 * spec.Rds_on * on).mean()),
        "inductor_dcr": float((iL[s] ** 2 * spec.R_L).mean()),
        "diode": float((iL[s] * spec.V_f * off).mean()),
        "esr": float((ic ** 2 * spec.ESR).mean()),
    }
    return res


def ripple_formulas(spec: ConverterSpec):
    """Textbook CCM ripple: returns (delta_iL_pp, delta_vout_pp) for the ideal converter."""
    D, T, L, C, R = spec.D, spec.T, spec.L, spec.C, spec.R
    Vo = spec.ideal_vout()
    if spec.topology == "buck":
        di = (spec.Vin - Vo) * D * T / L
        dv = di * T / (8 * C)
    else:
        di = spec.Vin * D * T / L
        dv = Vo * D * T / (R * C)
    return di, dv
