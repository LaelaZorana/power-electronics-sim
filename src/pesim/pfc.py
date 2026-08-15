"""Boost PFC with average current mode control (averaged model, no switching ripple).

The inductor current reference is k * |v_in(t)|. A PI current loop drives the
average duty cycle so the inductor current tracks the rectified sine. The
output capacitor sees the current (1 - d) iL and supplies the load, which
gives the familiar twice-line-frequency voltage ripple. A slow outer voltage PI
scales k. Non-ideal input current shape comes from finite current loop
bandwidth and the zero crossing distortion where the boost cannot regulate
(v_in near zero).
"""
from __future__ import annotations
import numpy as np
from .inverter import thd, fundamental


def boost_pfc(Vac_rms=230.0, f_line=50.0, Vout_ref=400.0, Pout=1000.0,
              L=1e-3, C=470e-6, fsw=100e3, kp_i=None, ki_i=None,
              n_cycles=20, dt=None):
    """Simulate the averaged boost PFC. Returns dict of waveforms and metrics."""
    dt = dt or 1 / (fsw * 2)
    T = n_cycles / f_line
    t = np.arange(0, T, dt)
    R = Vout_ref ** 2 / Pout
    w = 2 * np.pi * f_line
    vin = np.abs(np.sqrt(2) * Vac_rms * np.sin(w * t))
    # current loop tuned for ~ fsw/10 crossover on plant Vout/(sL)
    kp_i = kp_i or (2 * np.pi * fsw / 10) * L / Vout_ref
    ki_i = ki_i or kp_i * 2 * np.pi * f_line * 10
    kp_v, ki_v = 2e-4, 2e-3   # slow outer loop, well below 100 Hz so it does not distort the current
    iL = 0.0; vC = Vout_ref; xi = 0.0; xv = 0.0
    k_ref = Pout / Vac_rms ** 2  # conductance feed-forward
    IL = np.empty_like(t); VO = np.empty_like(t); Dc = np.empty_like(t)
    for n, tt in enumerate(t):
        ev = Vout_ref - vC
        xv += ev * dt
        k = max(k_ref + kp_v * ev + ki_v * xv, 0.0)
        iref = k * vin[n]
        ei = iref - iL
        xi += ei * dt
        d = 1 - vin[n] / max(vC, 1.0) + (kp_i * ei + ki_i * xi)  # feed-forward plus PI
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
