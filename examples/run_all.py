"""Generate every figure in figures/. Run: python examples/run_all.py

Needs pesim installed, pip install -e . from the repo root."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pesim.converters import ConverterSpec, simulate
from pesim.design import (buck_small_signal, boost_small_signal, design_type3,
                          design_type2, phase_deg, ccm_boundary_load, boost_rhp_zero)
from pesim.inverter import spwm, svpwm, line_line, spectrum, thd, fundamental
from pesim.pfc import boost_pfc
from pesim.thermal import MosfetParams, mosfet_losses, junction_temperature

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG, exist_ok=True)
summary = {}


def buck_ccm_dcm():
    base = dict(topology="buck", Vin=12, D=0.4, fs=100e3, L=100e-6, C=100e-6)
    Rb = ccm_boundary_load(ConverterSpec(**base, R=1))
    fig, ax = plt.subplots(2, 2, figsize=(10, 6), sharex="col")
    for j, (R, label) in enumerate([(0.5 * Rb, "CCM"), (2.5 * Rb, "DCM")]):
        r = simulate(ConverterSpec(**base, R=R), n_periods=3)
        t = r.t * 1e6
        ax[0, j].plot(t, r.iL); ax[0, j].set_title(f"{label}: R = {R:.1f} ohm, Vout = {r.Vout_avg:.2f} V")
        ax[0, j].set_ylabel("iL (A)")
        ax[1, j].plot(t, r.vout); ax[1, j].set_ylabel("vout (V)"); ax[1, j].set_xlabel("t (us)")
        summary[f"buck_{label}_Vout"] = r.Vout_avg
    fig.suptitle("Buck 12 V, D = 0.4, 100 kHz, L = 100 uH, C = 100 uF")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "buck_ccm_vs_dcm.png"), dpi=130); plt.close(fig)


def boost_efficiency():
    # switched sim gives conduction and diode losses, switching loss is added
    # from mosfet_losses fed with the simulated switch currents, so the curve
    # has the real shape: switching dominated at light load, conduction at heavy
    mos = MosfetParams(Rds_on_25=30e-3, Qg=40e-9, Qgd=8e-9, Qgs=7e-9, Qoss=60e-9,
                       Qrr=50e-9, Rth_jc=0.6, Rth_ca=8.0)
    loads = np.logspace(np.log10(3), np.log10(200), 25)
    eta, mode, Pout = [], [], []
    for R in loads:
        spec = ConverterSpec("boost", Vin=12, D=0.5, fs=100e3, L=100e-6, C=100e-6, R=R,
                             Rds_on=0.03, R_L=0.02, V_f=0.5, ESR=0.05)
        r = simulate(spec, n_periods=6)
        s = r.last_periods(5)
        on = r.sw[s] == 1
        i_on = float(r.iL[s][on][0]) if on.any() else 0.0    # current at turn on
        i_off = float(r.iL[s][on][-1]) if on.any() else 0.0  # current at turn off
        lm = mosfet_losses(mos, 0.0, i_on, i_off, Vds=r.Vout_avg, fsw=spec.fs)
        P_sw = lm["switching"] + lm["coss"] + lm["reverse_recovery"]
        eta.append(r.Pout / (r.Pin + P_sw)); mode.append(r.mode); Pout.append(r.Pout)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(Pout, np.array(eta) * 100, "o-")
    for p, m, e in zip(Pout, mode, eta):
        if m == "DCM":
            ax.plot(p, e * 100, "rs", ms=4)
    ax.set_xlabel("Pout (W), from the simulated output voltage"); ax.set_ylabel("efficiency (%)")
    ax.set_title("Boost 12 to 24 V, conduction plus switching loss, red = DCM")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "boost_efficiency_vs_load.png"), dpi=130); plt.close(fig)
    summary["boost_eta_max"] = max(eta)
    summary["boost_eta_peak_Pout"] = Pout[int(np.argmax(eta))]


def control_bode():
    spec = ConverterSpec("buck", Vin=12, D=0.4, fs=100e3, L=100e-6, C=100e-6, R=5, ESR=0.02)
    G = buck_small_signal(spec)
    Gc, m = design_type3(G, fc=5e3, pm_target=60)
    T = Gc * G
    w = 2 * np.pi * np.logspace(1, 5, 600)
    fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    for name, H in [("plant Gvd", G), ("compensator Gc", Gc), ("loop T", T)]:
        ax[0].semilogx(w / 2 / np.pi, 20 * np.log10(np.abs(H(1j * w))), label=name)
        ax[1].semilogx(w / 2 / np.pi, phase_deg(H, w), label=name)
    ax[0].axhline(0, color="k", lw=0.5); ax[0].set_ylabel("|H| (dB)"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].set_ylabel("phase (deg)"); ax[1].set_xlabel("f (Hz)"); ax[1].grid(alpha=0.3)
    ax[0].set_title(f"Buck voltage loop, type III: fc = {m['crossover_hz']:.0f} Hz, PM = {m['phase_margin_deg']:.1f} deg")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "control_loop_bode.png"), dpi=130); plt.close(fig)
    summary["buck_type3"] = m
    bspec = ConverterSpec("boost", Vin=12, D=0.5, fs=100e3, L=100e-6, C=100e-6, R=20)
    Gb = boost_small_signal(bspec)
    fz = boost_rhp_zero(bspec) / 2 / np.pi
    Gc2, m2 = design_type3(Gb, fc=fz / 5, pm_target=45)
    summary["boost_rhp_zero_hz"] = fz
    summary["boost_type3"] = m2


def inverter_figs():
    Vdc, m = 400, 0.8
    w = spwm(Vdc, m, f1=50, fsw=3e3, n_periods=1)
    vll = line_line(w)
    fig, ax = plt.subplots(3, 1, figsize=(9, 8))
    ax[0].plot(w["t"] * 1e3, w["refs"][0], w["t"] * 1e3, w["carrier"], lw=0.7)
    ax[0].set_title("SPWM reference and carrier (m = 0.8, fsw = 3 kHz)")
    ax[1].plot(w["t"] * 1e3, vll, lw=0.7); ax[1].set_ylabel("Vab (V)"); ax[1].set_xlabel("t (ms)")
    f, a = spectrum(vll, w["t"], 50, n_harm=140)
    ax[2].stem(f, a, basefmt=" ", markerfmt=" ")
    ax[2].set_xlabel("f (Hz)"); ax[2].set_ylabel("|Vab| peak (V)")
    ax[2].set_title(f"Line-line spectrum: fundamental {a[1]:.1f} V, theory {np.sqrt(3)*m*Vdc/2:.1f} V, unfiltered voltage THD {100*thd(vll, w['t'], 50):.0f} %")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "inverter_waveforms_fft.png"), dpi=130); plt.close(fig)
    summary["spwm_Vll1"] = a[1]
    ms = np.linspace(0.1, 1.155, 27)  # sweep stops at the SVPWM linear limit
    v_sp, v_sv, thd_sp, thd_sv = [], [], [], []
    for mi in ms:
        for gen, V, T in [(spwm, v_sp, thd_sp), (svpwm, v_sv, thd_sv)]:
            ww = gen(Vdc, mi, f1=50, fsw=5e3, n_periods=1)
            x = line_line(ww)
            V.append(fundamental(x, ww["t"], 50) / Vdc); T.append(100 * thd(x, ww["t"], 50))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ms, v_sp, label="SPWM"); ax[0].plot(ms, v_sv, label="SVPWM (min-max injection)")
    ax[0].axhline(np.sqrt(3) / 2, ls="--", color="gray", label="0.866, SPWM linear limit")
    ax[0].axhline(1.0, ls=":", color="gray", label="1.0, SVPWM linear limit")
    ax[0].set_xlabel("modulation index m"); ax[0].set_ylabel("Vll,1 peak / Vdc"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[0].set_title("DC bus utilisation")
    ax[1].plot(ms, thd_sp, label="SPWM"); ax[1].plot(ms, thd_sv, label="SVPWM")
    ax[1].set_xlabel("m"); ax[1].set_ylabel("unfiltered line-line voltage THD (%)"); ax[1].legend(); ax[1].grid(alpha=0.3)
    ax[1].set_title("Harmonic distortion vs m")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "svpwm_vs_spwm_utilisation.png"), dpi=130); plt.close(fig)
    summary["spwm_max_util"] = max(v_sp); summary["svpwm_max_util"] = max(v_sv)


def pfc_fig():
    r = boost_pfc(n_cycles=12)
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    ax[0].plot(r["t_ac"] * 1e3, r["iin_ac"]); ax[0].set_ylabel("input current (A)")
    ax[0].set_title(f"Boost PFC 230 V / 400 V / 1 kW: THD {100*r['thd']:.1f} %, PF {r['pf']:.4f}")
    ax[1].plot(r["t"] * 1e3, r["vout"]); ax[1].set_ylabel("Vout (V)"); ax[1].set_xlabel("t (ms)")
    ax[1].set_title(f"Output ripple {r['vout_ripple_pp']:.1f} Vpp at 100 Hz")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "pfc_input_current.png"), dpi=130); plt.close(fig)
    summary["pfc_thd"] = r["thd"]; summary["pfc_pf"] = r["pf"]


def thermal_fig():
    p = MosfetParams(Rds_on_25=10e-3, Qg=50e-9, Qgd=10e-9, Qgs=8e-9, Qoss=40e-9, Qrr=30e-9,
                     Rth_jc=0.5, Rth_ca=4.0)
    fs = np.logspace(4, 6, 40)
    Tj, Pc, Ps = [], [], []
    for f in fs:
        r = junction_temperature(p, I_rms=8, I_on=10, I_off=10, Vds=48, fsw=f, T_amb=40)
        Tj.append(r["Tj"]); Pc.append(r["conduction"]); Ps.append(r["switching"] + r["coss"] + r["reverse_recovery"])
    fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax[0].semilogx(fs, Pc, label="conduction"); ax[0].semilogx(fs, Ps, label="switching + Coss + Qrr")
    ax[0].set_ylabel("loss (W)"); ax[0].legend(loc="upper left"); ax[0].grid(alpha=0.3)
    ax[0].set_title("MOSFET losses and junction temperature vs switching frequency")
    ax[1].semilogx(fs, Tj, "k-"); ax[1].set_ylabel("Tj (C)"); ax[1].set_xlabel("fsw (Hz)"); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "mosfet_thermal.png"), dpi=130); plt.close(fig)
    summary["Tj_100k"] = junction_temperature(p, 8, 10, 10, 48, 100e3, 40)["Tj"]


if __name__ == "__main__":
    buck_ccm_dcm(); boost_efficiency(); control_bode(); inverter_figs(); pfc_fig(); thermal_fig()
    for k, v in summary.items():
        print(k, v)
