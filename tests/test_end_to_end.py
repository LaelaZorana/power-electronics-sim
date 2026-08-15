import numpy as np
from pesim.converters import ConverterSpec, simulate
from pesim.design import size_inductor, size_capacitor, control_to_output, design_type3
from pesim.devices import data_dir, load_all_devices, load_design, efficiency_curve


def test_buck_design_end_to_end():
    """Load the 48 to 12 V design, size it, simulate it, close the loop, and
    check the efficiency curve from the device data, the full data to result path."""
    design = load_design(data_dir() / "designs" / "buck-48to12-300w.yaml")
    dev = load_all_devices()[design["device"]]
    Vin, Vout, Pout, fsw = (float(design[k]) for k in ("Vin", "Vout", "Pout", "fsw"))
    R = Vout ** 2 / Pout
    D = Vout / Vin
    base = ConverterSpec("buck", Vin=Vin, D=D, fs=fsw, L=1e-3, C=1e-3, R=R)
    L = size_inductor(base, float(design["ripple_current_ratio"]))
    C = size_capacitor(base, float(design["ripple_voltage_ratio"]), L=L)
    spec = ConverterSpec("buck", Vin=Vin, D=D, fs=fsw, L=L, C=C, R=R,
                         Rds_on=dev.rds_on(100))
    r = simulate(spec, n_periods=6, steps_per_period=400)
    assert r.mode == "CCM"
    assert abs(r.Vout_avg - 12.0) < 0.15                    # loss drops a little
    assert abs(r.di_pp / r.IL_avg - 0.3) < 0.03             # sized ripple hit
    Gc, m = design_type3(control_to_output(spec),
                         fc=float(design["loop"]["fc"]),
                         pm_target=float(design["loop"]["pm_target"]))
    assert abs(m["phase_margin_deg"] - 60) < 2
    assert m["gain_margin_dB"] > 6
    curve = efficiency_curve(dev, Vin, Vout, Pout, fsw, kind="buck", Tj=100)
    assert curve["efficiency"][-1] > 0.97                   # semiconductor only
    # at light load switching loss dominates, at full load conduction grows
    assert curve["switching"][0] > curve["conduction"][0]
    assert curve["conduction"][-1] / curve["switching"][-1] > \
        curve["conduction"][0] / curve["switching"][0]
