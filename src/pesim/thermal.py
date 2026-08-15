"""MOSFET conduction and switching loss estimate from datasheet-style parameters."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class MosfetParams:
    Rds_on_25: float          # ohms at 25 C
    Qg: float = 0.0           # gate charge, C (for driver loss)
    Qgd: float = 0.0          # gate-drain charge, C
    Qgs: float = 0.0          # gate-source charge, C
    Qoss: float = 0.0         # output charge, C
    Qrr: float = 0.0          # body diode reverse recovery, C
    Vgs_drive: float = 10.0
    Vplateau: float = 4.5
    Ig_on: float = 1.0        # gate current during on transition, A
    Ig_off: float = 1.0
    Rth_jc: float = 1.0       # K/W
    Rth_ca: float = 10.0      # K/W (heatsink + case)
    alpha: float = 0.004      # Rds_on temperature coefficient per K (approx 0.4 %/K)


def rds_on_at(p: MosfetParams, Tj: float) -> float:
    return p.Rds_on_25 * (1 + p.alpha * (Tj - 25.0))


def mosfet_losses(p: MosfetParams, I_rms: float, I_on: float, I_off: float,
                  Vds: float, fsw: float, Tj: float = 25.0, hard_switched=True) -> dict:
    """Return per-mechanism losses in W.

    Conduction: I_rms^2 * Rds_on(Tj).
    Switching: 0.5 Vds I t_r/f, with rise and fall times taken as the time to
    move Qgd (Miller charge) plus half of Qgs with the given gate current.
    Output capacitance loss 0.5 Qoss Vds fsw and reverse recovery Qrr Vds fsw
    are added for hard switching. Gate drive loss Qg Vgs fsw is reported but is
    dissipated in the driver, not the die.
    """
    Rds = rds_on_at(p, Tj)
    P_cond = I_rms ** 2 * Rds
    t_on = (p.Qgd + p.Qgs / 2) / p.Ig_on if hard_switched else 0.0
    t_off = (p.Qgd + p.Qgs / 2) / p.Ig_off if hard_switched else 0.0
    P_sw = 0.5 * Vds * (I_on * t_on + I_off * t_off) * fsw
    P_oss = 0.5 * p.Qoss * Vds * fsw if hard_switched else 0.0
    P_rr = p.Qrr * Vds * fsw if hard_switched else 0.0
    P_gate = p.Qg * p.Vgs_drive * fsw
    total = P_cond + P_sw + P_oss + P_rr
    return {"conduction": P_cond, "switching": P_sw, "coss": P_oss,
            "reverse_recovery": P_rr, "gate_drive": P_gate, "total": total,
            "Rds_on": Rds}


def junction_temperature(p: MosfetParams, I_rms, I_on, I_off, Vds, fsw,
                         T_amb=25.0, iters=50) -> dict:
    """Iterate Tj = Ta + P(Tj) (Rth_jc + Rth_ca) to a self-consistent operating point."""
    Tj = T_amb
    for _ in range(iters):
        losses = mosfet_losses(p, I_rms, I_on, I_off, Vds, fsw, Tj)
        Tj_new = T_amb + losses["total"] * (p.Rth_jc + p.Rth_ca)
        if abs(Tj_new - Tj) < 1e-3:
            Tj = Tj_new
            break
        Tj = 0.5 * Tj + 0.5 * Tj_new
    losses["Tj"] = Tj
    losses["Tc"] = T_amb + losses["total"] * p.Rth_ca
    return losses
