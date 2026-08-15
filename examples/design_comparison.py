"""Design, simulate and compare the four converter designs in data/designs/.

For the buck and the boost the script sizes L and C from the ripple ratios,
simulates the switched converter with the device Rds_on at the assumed 100 C
junction temperature, designs the voltage loop compensator and reads back its
margins, then computes an efficiency curve versus load from the device data,
conduction plus switching loss. The PFC design runs the averaged boost PFC
model and reports the current loop margins of the inner PI on the plant
Vout over s L. The inverter design is a single operating point, per switch
losses come from the device tables with the phase current split between the
two switches of each leg, and six switches carry the three phases.

The device loss model is deliberately simple, one main switch per converter
for the DC converters, synchronous rectification and magnetics losses are not
counted, so the efficiencies here isolate the semiconductor contribution.

Writes examples/design_comparison.md and prints the same table.
Run: python examples/design_comparison.py
"""
import os
import numpy as np

from pesim.converters import ConverterSpec, simulate
from pesim.design import (size_inductor, size_capacitor, control_to_output,
                          design_type3, design_pi, boost_rhp_zero)
from pesim.devices import data_dir, load_all_devices, load_design, efficiency_curve, device_loss
from pesim.pfc import boost_pfc
import control as ct

HERE = os.path.dirname(os.path.abspath(__file__))
TJ = 100.0  # assumed junction temperature for device loss interpolation


def run_dc(design, dev):
    """Buck or boost: size, simulate, design the loop, device efficiency curve."""
    kind = design["kind"]
    Vin, Vout, Pout, fsw = (float(design[k]) for k in ("Vin", "Vout", "Pout", "fsw"))
    R = Vout ** 2 / Pout
    D = Vout / Vin if kind == "buck" else 1 - Vin / Vout
    base = ConverterSpec(kind, Vin=Vin, D=D, fs=fsw, L=1e-3, C=1e-3, R=R)
    L = size_inductor(base, float(design["ripple_current_ratio"]))
    C = size_capacitor(base, float(design["ripple_voltage_ratio"]), L=L)
    spec = ConverterSpec(kind, Vin=Vin, D=D, fs=fsw, L=L, C=C, R=R,
                         Rds_on=dev.rds_on(TJ))
    r = simulate(spec, n_periods=6, steps_per_period=400)
    G = control_to_output(spec)
    loop = design.get("loop", {})
    if "fc" in loop:
        fc = float(loop["fc"])
    else:
        fc = float(loop["fc_over_rhp_zero"]) * boost_rhp_zero(spec) / (2 * np.pi)
    Gc, m = design_type3(G, fc=fc, pm_target=float(loop.get("pm_target", 60)))
    curve = efficiency_curve(dev, Vin, Vout, Pout, fsw, kind=kind, Tj=TJ)
    return {"eta_full": float(curve["efficiency"][-1]),
            "eta_half": float(np.interp(0.5, np.linspace(0.1, 1.0, 19), curve["efficiency"])),
            "ripple": f"di {r.di_pp:.2f} A, dv {r.dv_pp * 1e3:.0f} mV",
            "margins": f"fc {m['crossover_hz']:.0f} Hz, PM {m['phase_margin_deg']:.1f} deg, "
                       f"GM {m['gain_margin_dB']:.1f} dB",
            "detail": f"L {L * 1e6:.0f} uH, C {C * 1e6:.0f} uF, Vout {r.Vout_avg:.2f} V, {r.mode}"}


def run_pfc(design, dev):
    Vac, Vout, Pout, fsw = (float(design[k]) for k in ("Vac_rms", "Vout", "Pout", "fsw"))
    L, C = float(design["L"]), float(design["C"])
    r = boost_pfc(Vac_rms=Vac, Vout_ref=Vout, Pout=Pout, L=L, C=C, fsw=fsw, n_cycles=12)
    # inner current loop margins on the plant Vout/(sL) with the same crossover target
    plant = ct.tf([Vout], [L, 0])
    _, m = design_pi(plant, fc=fsw / 10)
    # device losses: switch conducts for duty d of each step, boost switched voltage is Vout
    d = r["duty"]; iL = r["iL"]
    i_rms_sw = float(np.sqrt(np.mean(iL ** 2 * d)))
    i_avg = float(np.mean(iL))
    loss = device_loss(dev, i_rms_sw, i_avg, Vout, fsw, Tj=TJ)
    loss_half = device_loss(dev, i_rms_sw / 2, i_avg / 2, Vout, fsw, Tj=TJ)
    eta = Pout / (Pout + loss["total"])
    eta_half = (Pout / 2) / (Pout / 2 + loss_half["total"])
    return {"eta_full": eta, "eta_half": eta_half,
            "ripple": f"Vout {r['vout_ripple_pp']:.1f} Vpp at twice line frequency",
            "margins": f"current loop fc {m['crossover_hz']:.0f} Hz, PM {m['phase_margin_deg']:.1f} deg",
            "detail": f"THD {100 * r['thd']:.1f} %, PF {r['pf']:.3f}"}


def run_inverter(design, dev):
    Vdc, Pout, fsw = (float(design[k]) for k in ("Vdc", "Pout", "fsw"))
    m_i, pf = float(design["m"]), float(design["power_factor"])
    Vll1_rms = np.sqrt(3) * m_i * Vdc / 2 / np.sqrt(2)
    I_ph = Pout / (np.sqrt(3) * Vll1_rms * pf)
    i_rms_sw = I_ph / np.sqrt(2)   # each switch conducts half the fundamental period
    loss_sw = device_loss(dev, i_rms_sw, I_ph, Vdc, fsw, Tj=TJ)
    total = 6 * loss_sw["total"]
    half = 6 * device_loss(dev, i_rms_sw / 2, I_ph / 2, Vdc, fsw, Tj=TJ)["total"]
    return {"eta_full": Pout / (Pout + total),
            "eta_half": (Pout / 2) / (Pout / 2 + half),
            "ripple": "switched pole voltage, filtered by the machine inductance",
            "margins": "open loop operating point, no regulator in this model",
            "detail": f"I_phase {I_ph:.1f} A rms, per switch loss {loss_sw['total']:.1f} W"}


def main():
    devices = load_all_devices()
    rows = []
    for fname in sorted((data_dir() / "designs").glob("*.yaml")):
        design = load_design(fname)
        dev = devices[design["device"]]
        run = {"buck": run_dc, "boost": run_dc, "pfc": run_pfc,
               "inverter": run_inverter}[design["kind"]]
        res = run(design, dev)
        rows.append((design["name"], dev.name, res))
    lines = ["# Design comparison", "",
             "Generated by examples/design_comparison.py from data/designs/ and data/devices/.",
             "Efficiencies count semiconductor conduction and switching loss from the device",
             "tables at an assumed junction temperature of 100 C.", "",
             "| Design | Device | Eff full load | Eff half load | Ripple | Loop margins | Detail |",
             "|---|---|---|---|---|---|---|"]
    for name, devname, r in rows:
        lines.append(f"| {name} | {devname} | {100 * r['eta_full']:.2f} % | "
                     f"{100 * r['eta_half']:.2f} % | {r['ripple']} | {r['margins']} | {r['detail']} |")
    text = "\n".join(lines) + "\n"
    out = os.path.join(HERE, "design_comparison.md")
    with open(out, "w") as f:
        f.write(text)
    print(text)
    print("wrote", out)


if __name__ == "__main__":
    main()
