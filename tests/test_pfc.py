import numpy as np
import pytest
from pesim.pfc import boost_pfc
from pesim.inverter import fundamental


def test_pfc_baseline_metrics():
    # the README quotes 4.5 percent THD and PF 0.998 for the defaults
    r = boost_pfc(n_cycles=12)
    assert abs(r["thd"] - 0.045) < 0.003
    assert r["pf"] > 0.997
    # capacitor sees a 100 Hz current of amplitude Io, so the peak to peak
    # ripple is 2 Io / (w C) = 2 * 2.5 / (628 * 470 uF) = 16.9 Vpp
    Io = 1000.0 / 400.0
    expected = 2 * Io / (2 * np.pi * 100 * 470e-6)
    assert abs(r["vout_ripple_pp"] - expected) / expected < 0.05


def test_pfc_thd_mechanism_is_outer_loop_ripple():
    # the residual THD is 3rd harmonic injected by the outer voltage loop's
    # proportional gain multiplying the twice line frequency output ripple
    # into the current reference, not the current loop bandwidth
    base = boost_pfc(n_cycles=12)
    I3 = fundamental(base["iin_ac"], base["t_ac"], 3 * 50.0)
    assert I3 / base["I1_peak"] > 0.03          # 3rd harmonic dominates the residual
    no_kpv = boost_pfc(n_cycles=12, kp_v=0.0)
    assert no_kpv["thd"] < 0.005                # kills the distortion
    assert no_kpv["pf"] > 0.9999
    L, fsw, Vout = 1e-3, 100e3, 400.0
    kp_i_default = (2 * np.pi * fsw / 10) * L / Vout
    tripled = boost_pfc(n_cycles=12, kp_i=3 * kp_i_default)
    assert abs(tripled["thd"] - base["thd"]) < 0.005  # current loop gain is not the cause


def test_pfc_zero_gain_is_respected():
    # kp_i = 0 must not silently fall back to the default
    r = boost_pfc(n_cycles=4, kp_i=0.0, ki_i=0.0)
    assert r["thd"] > 0.2   # feed-forward only tracking is visibly worse
