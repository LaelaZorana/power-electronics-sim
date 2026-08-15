import numpy as np
import pytest
from pesim.converters import ConverterSpec, simulate, ripple_formulas
from pesim.design import (ccm_boundary_load, boost_rhp_zero, boost_small_signal,
                          buck_small_signal, design_type3, design_pi)
from pesim.inverter import spwm, svpwm, line_line, thd, fundamental, spectrum
from pesim.thermal import MosfetParams, junction_temperature


def test_buck_steady_state_vout():
    spec = ConverterSpec("buck", Vin=12, D=0.4, fs=100e3, L=100e-6, C=100e-6, R=5)
    r = simulate(spec)
    assert r.mode == "CCM"
    assert abs(r.Vout_avg - 0.4 * 12) < 0.01 * 4.8


def test_boost_steady_state_vout():
    spec = ConverterSpec("boost", Vin=12, D=0.5, fs=100e3, L=100e-6, C=100e-6, R=20)
    r = simulate(spec)
    assert r.mode == "CCM"
    assert abs(r.Vout_avg - 24) < 0.24


def test_buck_ripple_formulas():
    spec = ConverterSpec("buck", Vin=12, D=0.4, fs=100e3, L=100e-6, C=100e-6, R=5)
    r = simulate(spec, steps_per_period=400)
    di, dv = ripple_formulas(spec)
    assert abs(r.di_pp - di) / di < 0.03
    assert abs(r.dv_pp - dv) / dv < 0.05


def test_boost_ripple_formulas():
    spec = ConverterSpec("boost", Vin=12, D=0.5, fs=100e3, L=100e-6, C=100e-6, R=20)
    r = simulate(spec, steps_per_period=400)
    di, dv = ripple_formulas(spec)
    assert abs(r.di_pp - di) / di < 0.03
    assert abs(r.dv_pp - dv) / dv < 0.05


def test_ccm_boundary():
    spec = ConverterSpec("buck", Vin=12, D=0.4, fs=100e3, L=100e-6, C=100e-6, R=5)
    Rb = ccm_boundary_load(spec)   # 2L/((1-D)T) = 33.3 ohm
    assert abs(Rb - 2 * 100e-6 / (0.6 * 10e-6)) < 1e-9
    r_ccm = simulate(ConverterSpec("buck", 12, 0.4, 100e3, 100e-6, 100e-6, R=0.8 * Rb))
    r_dcm = simulate(ConverterSpec("buck", 12, 0.4, 100e3, 100e-6, 100e-6, R=1.25 * Rb))
    assert r_ccm.mode == "CCM" and r_dcm.mode == "DCM"
    assert r_dcm.Vout_avg > 4.8  # DCM raises the conversion ratio


def test_boost_rhp_zero():
    spec = ConverterSpec("boost", Vin=12, D=0.5, fs=100e3, L=100e-6, C=100e-6, R=20)
    G = boost_small_signal(spec)
    zeros = G.zeros()
    rhp = zeros[zeros.real > 0]
    assert len(rhp) == 1
    assert abs(rhp[0].real - boost_rhp_zero(spec)) / boost_rhp_zero(spec) < 1e-9
    assert abs(boost_rhp_zero(spec) - 0.25 * 20 / 100e-6) < 1e-6


def test_buck_dc_gain_and_resonance():
    spec = ConverterSpec("buck", Vin=12, D=0.4, fs=100e3, L=100e-6, C=100e-6, R=5)
    G = buck_small_signal(spec)
    assert abs(G(0j) - 12) < 1e-9
    w0 = 1 / np.sqrt(100e-6 * 100e-6)
    poles = G.poles()
    assert abs(abs(poles[0]) - w0) / w0 < 1e-9


def test_type3_hits_targets():
    spec = ConverterSpec("buck", Vin=12, D=0.4, fs=100e3, L=100e-6, C=100e-6, R=5)
    G = buck_small_signal(spec)
    Gc, m = design_type3(G, fc=5e3, pm_target=60)
    assert abs(m["crossover_hz"] - 5e3) / 5e3 < 0.02
    assert abs(m["phase_margin_deg"] - 60) < 2


def test_spwm_fundamental():
    Vdc = 400
    for m in (0.4, 0.8, 1.0):
        w = spwm(Vdc, m, f1=50, fsw=10e3, n_periods=2)
        V1 = fundamental(w["va"], w["t"], 50)
        assert abs(V1 - m * Vdc / 2) / (m * Vdc / 2) < 0.02
        Vll = fundamental(line_line(w), w["t"], 50)
        assert abs(Vll - np.sqrt(3) * m * Vdc / 2) / (np.sqrt(3) * m * Vdc / 2) < 0.02


def test_svpwm_extends_linear_range():
    Vdc = 400
    m = 1.15
    w = svpwm(Vdc, m, f1=50, fsw=10e3, n_periods=2)
    V1 = fundamental(w["va"], w["t"], 50)
    assert abs(V1 - m * Vdc / 2) / (m * Vdc / 2) < 0.02   # still linear at m = 1.15
    ws = spwm(Vdc, m, f1=50, fsw=10e3, n_periods=2)
    assert fundamental(ws["va"], ws["t"], 50) < 0.99 * m * Vdc / 2  # SPWM has saturated


def test_square_wave_thd():
    t = np.arange(0, 1.0, 1 / 20000)
    sq = np.sign(np.sin(2 * np.pi * 5 * t))
    val = thd(sq, t, 5)
    assert abs(val - np.sqrt(np.pi ** 2 / 8 - 1)) < 0.005   # 48.34 %
    f, a = spectrum(sq, t, 5, n_harm=5)
    assert abs(a[1] - 4 / np.pi) < 0.01 and abs(a[3] - 4 / (3 * np.pi)) < 0.01


def test_thermal_converges():
    p = MosfetParams(Rds_on_25=10e-3, Qg=50e-9, Qgd=10e-9, Qgs=8e-9, Qoss=40e-9,
                     Rth_jc=0.5, Rth_ca=5.0)
    r = junction_temperature(p, I_rms=8, I_on=10, I_off=10, Vds=48, fsw=100e3, T_amb=40)
    assert r["Tj"] > 40 and r["Tj"] < 175
    assert abs(r["Tj"] - (40 + r["total"] * 5.5)) < 0.01
