"""MOSFET conduction and switching loss estimate from datasheet-style parameters."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class MosfetParams:
    Rds_on_25: float          # ohms at 25 C
    Qg: float = 0.0           # gate charge, C, for driver loss
    Qgd: float = 0.0          # gate-drain charge, C
    Qgs: float = 0.0          # gate-source charge, C
    Qoss: float = 0.0         # output charge, C
    Qrr: float = 0.0          # body diode reverse recovery, C
    Vgs_drive: float = 10.0
    Ig_on: float = 1.0        # gate current during on transition, A
    Ig_off: float = 1.0
    Rth_jc: float = 1.0       # K/W
    Rth_ca: float = 10.0      # K/W, case to ambient including any heatsink
    alpha: float = 0.004      # Rds_on temperature coefficient per K. 0.004 gives
                              # 1.5x at 150 C, the low side of typical Si which
                              # runs 1.7 to 2x, so treat it as a device parameter.


def rds_on_at(p: MosfetParams, Tj: float) -> float:
    return p.Rds_on_25 * (1 + p.alpha * (Tj - 25.0))


def mosfet_losses(p: MosfetParams, I_rms: float, I_on: float, I_off: float,
                  Vds: float, fsw: float, Tj: float = 25.0, hard_switched=True) -> dict:
    """Return per-mechanism losses in W.

    Conduction: I_rms^2 * Rds_on(Tj).
    Switching: 0.5 Vds I t_r fsw, with rise and fall times taken as the time to
    move Qgd, the Miller charge, plus half of Qgs with the given gate current.
    Output capacitance loss 0.5 Qoss Vds fsw and reverse recovery Qrr Vds fsw
    are added for hard switching. The Coss term overestimates the stored energy
    for a nonlinear Coss, where Eoss is below 0.5 Qoss Vds. Gate drive loss
    Qg Vgs fsw is reported but is dissipated in the driver, not the die.
    """
    if p.Ig_on <= 0 or p.Ig_off <= 0:
        raise ValueError("gate currents Ig_on and Ig_off must be positive")
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
                         T_amb=25.0) -> dict:
    """Solve Tj = Ta + P(Tj) (Rth_jc + Rth_ca) exactly.

    Total loss is affine in Tj because only conduction loss depends on
    temperature and Rds_on is linear in Tj, so the self-consistent junction
    temperature has a closed form. When the temperature feedback gain
    alpha * I_rms^2 * Rds_on_25 * (Rth_jc + Rth_ca) reaches 1 the operating
    point has no finite solution, which is thermal runaway, and a ValueError
    is raised instead of returning a divergent number.
    """
    Rth = p.Rth_jc + p.Rth_ca
    b = I_rms ** 2 * p.Rds_on_25   # conduction loss at the alpha reference point
    gain = p.alpha * b * Rth
    if gain >= 1.0:
        raise ValueError(
            f"thermal runaway: feedback gain alpha*Irms^2*Rds25*Rth = {gain:.3f} >= 1")
    l25 = mosfet_losses(p, I_rms, I_on, I_off, Vds, fsw, Tj=25.0)
    P_fixed = l25["total"] - b  # switching, Coss and Qrr do not depend on Tj
    # Tj = Ta + (P_fixed + b (1 + alpha (Tj - 25))) Rth, solve for Tj
    Tj = (T_amb + (P_fixed + b * (1 - 25 * p.alpha)) * Rth) / (1.0 - gain)
    losses = mosfet_losses(p, I_rms, I_on, I_off, Vds, fsw, Tj=Tj)
    losses["Tj"] = Tj
    losses["Tc"] = T_amb + losses["total"] * p.Rth_ca
    return losses
