import warnings

import numpy as np
import pytest
from pesim.converters import ConverterSpec, simulate, ripple_formulas
from pesim.design import (ccm_boundary_load, ccm_boundary_inductance, boost_rhp_zero,
                          boost_small_signal, buck_small_signal, buckboost_small_signal,
                          design_type3, design_type2, design_pi,
                          size_inductor, size_capacitor)
from pesim.inverter import (spwm, svpwm, line_line, thd, fundamental, spectrum,
                            dc_utilisation, dead_time_voltage_error)
from pesim.thermal import MosfetParams, mosfet_losses, junction_temperature


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


def test_buckboost_steady_state_and_ripple():
    spec = ConverterSpec("buckboost", Vin=12, D=0.4, fs=100e3, L=100e-6, C=100e-6, R=3)
    r = simulate(spec, steps_per_period=400)
    assert r.mode == "CCM"
    assert abs(r.Vout_avg - 8.0) < 0.01 * 8.0        # D/(1-D) * 12 = 8, magnitude
    assert abs(r.IL_avg - 8.0 / 3 / 0.6) < 0.02 * 4.444  # Io/(1-D)
    di, dv = ripple_formulas(spec)
    assert abs(di - 0.48) < 1e-9 and abs(dv - 0.32 / 3) < 1e-9
    assert abs(r.di_pp - di) / di < 0.03
    assert abs(r.dv_pp - dv) / dv < 0.05


def test_buckboost_ccm_boundary():
    spec = ConverterSpec("buckboost", Vin=12, D=0.4, fs=100e3, L=100e-6, C=100e-6, R=3)
    Rb = ccm_boundary_load(spec)   # 2L/((1-D)^2 T) = 55.6 ohm
    assert abs(Rb - 2 * 100e-6 / (0.36 * 10e-6)) < 1e-9
    kw = dict(fs=100e3, L=100e-6, C=100e-6)
    assert simulate(ConverterSpec("buckboost", 12, 0.4, R=0.9 * Rb, **kw)).mode == "CCM"
    assert simulate(ConverterSpec("buckboost", 12, 0.4, R=1.1 * Rb, **kw)).mode == "DCM"


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


def test_esr_ripple_step():
    ideal = ConverterSpec("boost", Vin=12, D=0.5, fs=100e3, L=100e-6, C=100e-6, R=20)
    esr = ConverterSpec("boost", Vin=12, D=0.5, fs=100e3, L=100e-6, C=100e-6, R=20, ESR=0.05)
    r0 = simulate(ideal, steps_per_period=400)
    r1 = simulate(esr, steps_per_period=400)
    # the ESR adds roughly ESR * (IL + Io) of step to the ripple at the switching edge
    assert r1.dv_pp > r0.dv_pp + 0.5 * 0.05 * r1.IL_avg


def test_ccm_boundary():
    spec = ConverterSpec("buck", Vin=12, D=0.4, fs=100e3, L=100e-6, C=100e-6, R=5)
    Rb = ccm_boundary_load(spec)   # 2L/((1-D)T) = 33.3 ohm
    assert abs(Rb - 2 * 100e-6 / (0.6 * 10e-6)) < 1e-9
    r_ccm = simulate(ConverterSpec("buck", 12, 0.4, 100e3, 100e-6, 100e-6, R=0.8 * Rb))
    r_dcm = simulate(ConverterSpec("buck", 12, 0.4, 100e3, 100e-6, 100e-6, R=1.25 * Rb))
    assert r_ccm.mode == "CCM" and r_dcm.mode == "DCM"
    assert r_dcm.Vout_avg > 4.8  # DCM raises the conversion ratio


def test_buck_dcm_example_voltage():
    # the README quotes 6.58 V for the buck at 2.5 times the boundary load
    base = dict(topology="buck", Vin=12, D=0.4, fs=100e3, L=100e-6, C=100e-6)
    Rb = ccm_boundary_load(ConverterSpec(**base, R=1))
    r = simulate(ConverterSpec(**base, R=2.5 * Rb), n_periods=3)
    assert r.mode == "DCM"
    assert abs(r.Vout_avg - 6.58) < 0.02


def test_boost_dcm_closed_form():
    # DCM boost voltage M = (1 + sqrt(1 + 4 D^2 / K)) / 2 with K = 2L/(R T)
    spec = ConverterSpec("boost", Vin=12, D=0.5, fs=100e3, L=100e-6, C=100e-6, R=200)
    r = simulate(spec, steps_per_period=400)
    K = 2 * spec.L / (spec.R * spec.T)
    M = (1 + np.sqrt(1 + 4 * spec.D ** 2 / K)) / 2
    assert r.mode == "DCM"
    assert abs(r.Vout_avg - 12 * M) / (12 * M) < 1e-3


def test_power_balance():
    spec = ConverterSpec("boost", Vin=12, D=0.5, fs=100e3, L=100e-6, C=100e-6, R=20,
                         Rds_on=0.03, R_L=0.02, V_f=0.5, ESR=0.05)
    r = simulate(spec, steps_per_period=400)
    residual = r.Pin - r.Pout - sum(r.losses.values())
    assert abs(residual) / r.Pin < 1e-3


def test_spec_validation():
    with pytest.raises(ValueError):
        ConverterSpec("buck", D=1.2)
    with pytest.raises(ValueError):
        ConverterSpec("buck", D=-0.1)
    with pytest.raises(ValueError):
        ConverterSpec("buck", Vin=-12)
    with pytest.raises(ValueError):
        ConverterSpec("buck", L=0)
    with pytest.raises(ValueError):
        ConverterSpec("flyback")
    with pytest.raises(ValueError):
        simulate(ConverterSpec("buck"), n_periods=0)


def test_duty_quantisation_warns_and_reports():
    spec = ConverterSpec("buck", Vin=12, D=0.333, fs=100e3, L=100e-6, C=100e-6, R=5)
    with pytest.warns(UserWarning, match="quantised"):
        r = simulate(spec, steps_per_period=20)
    assert abs(r.D_eff - 7 / 20) < 1e-12
    r2 = simulate(spec, steps_per_period=1000)
    assert abs(r2.D_eff - 0.333) < 1e-12


def test_sizing_round_trip():
    spec = ConverterSpec("buck", Vin=12, D=0.4, fs=100e3, L=100e-6, C=100e-6, R=5)
    L = size_inductor(spec, di_ratio=0.3)
    C = size_capacitor(spec, dv_ratio=0.01, L=L)
    sized = ConverterSpec("buck", Vin=12, D=0.4, fs=100e3, L=L, C=C, R=5)
    r = simulate(sized, steps_per_period=400)
    assert abs(r.di_pp / r.IL_avg - 0.3) < 0.02
    assert abs(r.dv_pp / r.Vout_avg - 0.01) < 0.002
    Lc = ccm_boundary_inductance(spec)
    assert abs(ccm_boundary_load(ConverterSpec("buck", 12, 0.4, 100e3, Lc, 100e-6, R=5))
               - spec.R) / spec.R < 1e-9


def test_boost_rhp_zero():
    spec = ConverterSpec("boost", Vin=12, D=0.5, fs=100e3, L=100e-6, C=100e-6, R=20)
    G = boost_small_signal(spec)
    zeros = G.zeros()
    rhp = zeros[zeros.real > 0]
    assert len(rhp) == 1
    assert abs(rhp[0].real - boost_rhp_zero(spec)) / boost_rhp_zero(spec) < 1e-9
    # independent check: the averaged inductor dynamics put the zero where the
    # step response of the output initially moves the wrong way, w_z = D'^2 R / L
    # = 0.25 * 20 / 1e-4 = 50 krad/s = 7957.7 Hz, pinned by hand
    assert abs(boost_rhp_zero(spec) - 50000.0) < 1e-6


def test_buck_dc_gain_and_resonance():
    spec = ConverterSpec("buck", Vin=12, D=0.4, fs=100e3, L=100e-6, C=100e-6, R=5)
    G = buck_small_signal(spec)
    assert abs(G(0j) - 12) < 1e-9
    w0 = 1 / np.sqrt(100e-6 * 100e-6)
    poles = G.poles()
    assert abs(abs(poles[0]) - w0) / w0 < 1e-9


def test_type3_hits_targets():
    # same plant as examples/run_all.py control_bode, including the 20 mohm ESR,
    # so the README table row and this test quote the same loop
    spec = ConverterSpec("buck", Vin=12, D=0.4, fs=100e3, L=100e-6, C=100e-6, R=5, ESR=0.02)
    G = buck_small_signal(spec)
    Gc, m = design_type3(G, fc=5e3, pm_target=60)
    assert abs(m["crossover_hz"] - 5e3) / 5e3 < 0.02
    assert abs(m["phase_margin_deg"] - 60) < 2
    assert abs(m["gain_margin_dB"] - 33.5) < 0.5


def test_type2_on_first_order_plant():
    import control as ct
    plant = ct.tf([100], [1e-3, 1])   # first order, easy for a type II
    Gc, m = design_type2(plant, fc=2e3, pm_target=60)
    assert abs(m["crossover_hz"] - 2e3) / 2e3 < 0.02
    assert abs(m["phase_margin_deg"] - 60) < 2


def test_design_pi_current_loop():
    import control as ct
    plant = ct.tf([400], [1e-3, 0])   # Vout/(sL), the PFC current loop plant
    Gc, m = design_pi(plant, fc=10e3)
    assert abs(m["crossover_hz"] - 10e3) / 10e3 < 0.02
    assert m["phase_margin_deg"] > 60


def test_design_raises_on_unachievable_or_unstable():
    spec = ConverterSpec("buck", Vin=12, D=0.4, fs=100e3, L=100e-6, C=100e-6, R=5)
    G = buck_small_signal(spec)
    with pytest.raises(ValueError):   # two pole plant needs 146 deg, type II caps at 89
        design_type2(G, fc=5e3, pm_target=60)
    with pytest.raises(ValueError):   # PI on a two pole plant above resonance is unstable
        design_pi(G, fc=5e3)
    bspec = ConverterSpec("boost", Vin=12, D=0.5, fs=100e3, L=100e-6, C=100e-6, R=20)
    Gb = boost_small_signal(bspec)
    with pytest.raises(ValueError):   # crossover below the resonance, degenerate k-factor
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            design_type3(Gb, fc=500, pm_target=45)


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


def test_utilisation_linear_limits():
    # linear range ends at 0.866 Vdc line to line for SPWM and 1.0 Vdc for SVPWM
    assert abs(dc_utilisation(1.0, 1.0, "spwm") - np.sqrt(3) / 2) < 1e-12
    assert abs(dc_utilisation(1.0, 2 / np.sqrt(3), "svpwm") - 1.0) < 1e-12
    assert dc_utilisation(1.0, 1.4, "spwm") == dc_utilisation(1.0, 1.0, "spwm")
    w = svpwm(400, 2 / np.sqrt(3), f1=50, fsw=10e3, n_periods=2)
    Vll = fundamental(line_line(w), w["t"], 50)
    assert abs(Vll / 400 - 1.0) < 0.02


def test_dead_time_voltage_error():
    # 400 V bus, 1 us dead time, 10 kHz gives an average 4 V error, by hand
    assert abs(dead_time_voltage_error(400, 1e-6, 10e3) - 4.0) < 1e-12


def test_square_wave_thd():
    t = np.arange(0, 1.0, 1 / 20000)
    sq = np.sign(np.sin(2 * np.pi * 5 * t))
    val = thd(sq, t, 5)
    assert abs(val - np.sqrt(np.pi ** 2 / 8 - 1)) < 0.005   # 48.34 %
    f, a = spectrum(sq, t, 5, n_harm=5)
    assert abs(a[1] - 4 / np.pi) < 0.01 and abs(a[3] - 4 / (3 * np.pi)) < 0.01


def test_thd_rejects_no_fundamental():
    t = np.arange(0, 1.0, 1 / 1000)
    with pytest.raises(ValueError):
        thd(np.zeros_like(t), t, 5)


def test_mosfet_losses_pinned():
    # hand computed: t_on = t_off = (10n + 4n)/1 A = 14 ns,
    # P_sw = 0.5 * 48 * (10 * 14n + 10 * 14n) * 100k = 0.672 W,
    # P_oss = 0.5 * 40n * 48 * 100k = 0.096 W,
    # P_cond at 25 C = 64 * 10 mohm = 0.640 W
    p = MosfetParams(Rds_on_25=10e-3, Qg=50e-9, Qgd=10e-9, Qgs=8e-9, Qoss=40e-9,
                     Rth_jc=0.5, Rth_ca=5.0)
    l = mosfet_losses(p, I_rms=8, I_on=10, I_off=10, Vds=48, fsw=100e3, Tj=25.0)
    assert abs(l["switching"] - 0.672) < 1e-9
    assert abs(l["coss"] - 0.096) < 1e-9
    assert abs(l["conduction"] - 0.640) < 1e-9


def test_thermal_self_consistent_pinned():
    # closed form by hand: b = 0.64 W, gain = 0.004 * 0.64 * 5.5 = 0.01408,
    # P_fixed = 0.768 W, Tj = (40 + (0.768 + 0.64 * 0.9) * 5.5) / (1 - 0.01408)
    # = 48.069 C
    p = MosfetParams(Rds_on_25=10e-3, Qg=50e-9, Qgd=10e-9, Qgs=8e-9, Qoss=40e-9,
                     Rth_jc=0.5, Rth_ca=5.0)
    r = junction_temperature(p, I_rms=8, I_on=10, I_off=10, Vds=48, fsw=100e3, T_amb=40)
    assert abs(r["Tj"] - 48.0688) < 1e-3
    # the fixed point equation itself must also hold
    assert abs(r["Tj"] - (40 + r["total"] * 5.5)) < 1e-6


def test_thermal_runaway_raises():
    p = MosfetParams(Rds_on_25=10e-3, Rth_jc=0.5, Rth_ca=100.0)
    with pytest.raises(ValueError, match="runaway"):
        junction_temperature(p, I_rms=20, I_on=20, I_off=20, Vds=48, fsw=100e3, T_amb=40)


def test_thermal_example_tj():
    # the README quotes Tj about 47 C for the example MOSFET at 100 kHz
    p = MosfetParams(Rds_on_25=10e-3, Qg=50e-9, Qgd=10e-9, Qgs=8e-9, Qoss=40e-9,
                     Qrr=30e-9, Rth_jc=0.5, Rth_ca=4.0)
    r = junction_temperature(p, 8, 10, 10, 48, 100e3, 40)
    assert abs(r["Tj"] - 47.24) < 0.05
