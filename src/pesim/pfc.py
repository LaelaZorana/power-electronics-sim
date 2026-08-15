"""Boost PFC with average current mode control, an averaged model with no switching ripple.

The inductor current reference is k * |v_in(t)|. A PI current loop drives the
average duty cycle so the inductor current tracks the rectified sine. The
output capacitor sees the current (1 - d) iL and supplies the load, which
gives the familiar twice-line-frequency voltage ripple. A slow outer voltage PI
scales k.

Where the residual distortion comes from: the twice-line-frequency output
ripple feeds back through the outer voltage loop's proportional gain kp_v and
modulates the current reference amplitude k, which injects an almost pure 3rd
harmonic into the input current. With the default kp_v the measured spectrum is
dominated by that 3rd harmonic and the THD is about 4.5 percent. Setting kp_v
to zero while keeping ki_v drops the THD below 0.5 percent with power factor
1.000, and tripling the current loop gain kp_i leaves the THD unchanged, so the
current loop bandwidth and zero crossing behaviour contribute almost nothing at
these settings. This is the textbook constraint that the voltage loop must
respond far below twice the line frequency, and its residual ripple gain is
what sets the THD. The test test_pfc_thd_mechanism pins this experiment.
"""
from __future__ import annotations
import numpy as np
from .inverter import thd, fundamental


def boost_pfc(Vac_rms=230.0, f_line=50.0, Vout_ref=400.0, Pout=1000.0,
              L=1e-3, C=470e-6, fsw=100e3, kp_i=None, ki_i=None,
              kp_v=2e-4, ki_v=2e-3, n_cycles=20, dt=None):
    """Simulate the averaged boost PFC. Returns dict of waveforms and metrics.

    kp_v and ki_v are the outer voltage loop gains in units of A per V of input
    voltage, per V of output error, and per V s of integrated error. The
    defaults give a loop far below 100 Hz but still slow enough that ripple
    feedthrough via kp_v dominates the input current THD, as described in the
    module docstring. The time step defaults to half a switching period, which
    for an averaged model is simply a fine Euler step, it is not tied to the
    switching instants.
    """
    if dt is None:
        dt = 1 / (fsw * 2)
    T = n_cycles / f_line
    t = np.arange(0, T, dt)
    R = Vout_ref ** 2 / Pout
    w = 2 * np.pi * f_line
    vin = np.abs(np.sqrt(2) * Vac_rms * np.sin(w * t))
    # current loop tuned for a crossover near fsw/10 on the plant Vout/(sL)
    if kp_i is None:
        kp_i = (2 * np.pi * fsw / 10) * L / Vout_ref
    if ki_i is None:
        ki_i = kp_i * 2 * np.pi * f_line * 10
    iL = 0.0; vC = Vout_ref; xi = 0.0; xv = 0.0
    k_ref = Pout / Vac_rms ** 2  # conductance feed-forward
    # starting at vC = Vout_ref skips the start-up transient on purpose, the
    # metrics are meant to describe cyclic steady state
    IL = np.empty_like(t); VO = np.empty_like(t); Dc = np.empty_like(t)
    for n, tt in enumerate(t):
        ev = Vout_ref - vC
        xv += ev * dt
        k = k_ref + kp_v * ev + ki_v * xv
        if k < 0.0:
            k = 0.0
            xv -= ev * dt  # anti windup on the outer integrator
        iref = k * vin[n]
        ei = iref - iL
        xi += ei * dt
        d = 1 - vin[n] / max(vC, 1.0) + (kp_i * ei + ki_i * xi)  # feed-forward plus PI
        if d > 1.0 or d < 0.0:
            xi -= ei * dt  # anti windup on the current integrator while d is clamped
            d = min(max(d, 0.0), 1.0)
        diL = (vin[n] - (1 - d) * vC) / L
        dvC = ((1 - d) * iL - vC / R) / C
        iL = max(iL + diL * dt, 0.0)
        vC += dvC * dt
        IL[n] = iL; VO[n] = vC; Dc[n] = d
    # metrics over the last 4 line cycles
    s = t >= T - 4 / f_line
    iin = IL[s] * np.sign(np.sin(w * t[s]))
    ts = t[s]
    thd_i = thd(iin, ts, f_line)
    I1 = fundamental(iin, ts, f_line)
    vin_ac = np.sqrt(2) * Vac_rms * np.sin(w * ts)
    P = np.mean(vin_ac * iin)
    S = Vac_rms * np.sqrt(np.mean(iin ** 2))
    return {"t": t, "vin": vin, "iL": IL, "vout": VO, "duty": Dc,
            "iin_ac": iin, "t_ac": ts, "thd": thd_i, "pf": P / S,
            "I1_peak": I1, "vout_ripple_pp": float(VO[s].max() - VO[s].min())}
