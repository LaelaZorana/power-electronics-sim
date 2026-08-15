"""Device parameter loading and datasheet style loss interpolation.

Devices live in data/devices/ as YAML files holding Rds_on versus junction
temperature points, total switching energy versus current points at a stated
reference voltage, thermal resistances and charge figures. The numbers are
representative of their datasheet class, not copies of a specific part, see
data/README.md. Converter design points live in data/designs/ as YAML files
that name a device and an operating point.

Loss model used by efficiency_curve: conduction loss is I_rms^2 times the
Rds_on interpolated at the assumed junction temperature, switching loss is the
per cycle energy interpolated at the switched current, scaled linearly with the
ratio of switched voltage to the reference voltage, times the switching
frequency. Both interpolations are linear with end point extrapolation clamped
to the tabulated range.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import yaml


@dataclass
class DeviceData:
    name: str
    technology: str
    Vds_max: float
    rds_on_vs_temp: np.ndarray      # rows of [Tj in C, Rds_on in ohm]
    esw_vs_current: np.ndarray      # rows of [current in A, Eon plus Eoff in J]
    V_ref_sw: float                 # voltage at which esw_vs_current was tabulated
    Rth_jc: float
    Rth_ca: float
    Qg: float = 0.0
    Qoss: float = 0.0
    Qrr: float = 0.0
    notes: str = ""

    def rds_on(self, Tj: float) -> float:
        """Rds_on in ohm at junction temperature Tj, linear interpolation."""
        pts = self.rds_on_vs_temp
        return float(np.interp(Tj, pts[:, 0], pts[:, 1]))

    def switching_energy(self, current: float, Vds: float | None = None) -> float:
        """Total Eon plus Eoff in J at the given current, scaled to Vds."""
        pts = self.esw_vs_current
        e = float(np.interp(current, pts[:, 0], pts[:, 1]))
        if Vds is not None:
            e *= Vds / self.V_ref_sw
        return e


def _as_points(rows, name):
    arr = np.asarray(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 2:
        raise ValueError(f"{name}: expected at least two [x, y] rows")
    if not np.all(np.diff(arr[:, 0]) > 0):
        raise ValueError(f"{name}: x values must be strictly increasing")
    return arr


def load_device(path) -> DeviceData:
    """Load one device YAML file into a DeviceData."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    required = ["name", "technology", "Vds_max", "rds_on_vs_temp",
                "esw_vs_current", "V_ref_sw", "Rth_jc", "Rth_ca"]
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"{path.name}: missing keys {missing}")
    return DeviceData(
        name=raw["name"], technology=raw["technology"],
        Vds_max=float(raw["Vds_max"]),
        rds_on_vs_temp=_as_points(raw["rds_on_vs_temp"], f"{path.name} rds_on_vs_temp"),
        esw_vs_current=_as_points(raw["esw_vs_current"], f"{path.name} esw_vs_current"),
        V_ref_sw=float(raw["V_ref_sw"]),
        Rth_jc=float(raw["Rth_jc"]), Rth_ca=float(raw["Rth_ca"]),
        Qg=float(raw.get("Qg", 0.0)), Qoss=float(raw.get("Qoss", 0.0)),
        Qrr=float(raw.get("Qrr", 0.0)), notes=str(raw.get("notes", "")))


def load_design(path) -> dict:
    """Load one converter design YAML file and validate its shared keys."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    required = ["name", "kind", "device", "Pout", "fsw"]
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"{path.name}: missing keys {missing}")
    kinds = ("buck", "boost", "pfc", "inverter")
    if raw["kind"] not in kinds:
        raise ValueError(f"{path.name}: kind must be one of {kinds}")
    return raw


def data_dir() -> Path:
    """Repo data directory, found relative to this file."""
    return Path(__file__).resolve().parents[2] / "data"


def load_all_devices(directory=None) -> dict[str, DeviceData]:
    directory = Path(directory) if directory else data_dir() / "devices"
    out = {}
    for p in sorted(directory.glob("*.yaml")):
        d = load_device(p)
        out[d.name] = d
    return out


def device_loss(dev: DeviceData, I_rms: float, I_sw: float, Vsw: float,
                fsw: float, Tj: float = 100.0) -> dict:
    """Conduction plus switching loss in W for one device at one operating point.

    I_rms sets conduction loss through Rds_on at Tj, I_sw is the current at the
    switching instants used to interpolate the tabulated energy, Vsw the
    switched voltage."""
    p_cond = I_rms ** 2 * dev.rds_on(Tj)
    p_sw = dev.switching_energy(I_sw, Vsw) * fsw
    return {"conduction": p_cond, "switching": p_sw, "total": p_cond + p_sw}


def efficiency_curve(dev: DeviceData, Vin: float, Vout: float, Pout_max: float,
                     fsw: float, kind: str = "buck", Tj: float = 100.0,
                     load_fractions=None) -> dict:
    """Efficiency versus load for a simple one device converter model.

    Uses the device Rds_on at Tj for conduction and the tabulated switching
    energy at the average switched current, scaled to the switched voltage.
    For a buck the switched voltage is Vin and the device carries the inductor
    current for the on fraction D. For a boost the switched voltage is Vout
    and the device conducts for D of the period. Returns arrays of load W,
    efficiency, and the per mechanism losses.
    """
    if load_fractions is None:
        load_fractions = np.linspace(0.1, 1.0, 19)
    load_fractions = np.asarray(load_fractions, dtype=float)
    P = load_fractions * Pout_max
    eta = np.empty_like(P); pc = np.empty_like(P); ps = np.empty_like(P)
    if kind == "buck":
        D = Vout / Vin
        Vsw = Vin
        I_dc = P / Vout                      # inductor and load current
    elif kind == "boost":
        D = 1 - Vin / Vout
        Vsw = Vout
        I_dc = P / Vin                       # inductor carries the input current
    else:
        raise ValueError("efficiency_curve handles kind 'buck' or 'boost'")
    for i, p_out in enumerate(P):
        i_dc = I_dc[i]
        i_rms = i_dc * np.sqrt(D)            # device conducts for D of the period
        loss = device_loss(dev, i_rms, i_dc, Vsw, fsw, Tj)
        pc[i] = loss["conduction"]; ps[i] = loss["switching"]
        eta[i] = p_out / (p_out + loss["total"])
    return {"Pout": P, "efficiency": eta, "conduction": pc, "switching": ps}
