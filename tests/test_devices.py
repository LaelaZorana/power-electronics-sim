import numpy as np
import pytest
from pesim.devices import (data_dir, load_device, load_design, load_all_devices,
                           device_loss, efficiency_curve)


def test_load_all_devices():
    devs = load_all_devices()
    assert set(devs) == {"sj-mosfet-650v", "sic-mosfet-650v",
                         "syncbuck-fet-100v", "sic-halfbridge-1200v"}
    for d in devs.values():
        assert d.Vds_max > 0 and d.Rth_jc > 0 and d.Rth_ca > 0
        # Rds_on must rise with temperature for every device
        assert d.rds_on(150) > d.rds_on(25)
        # switching energy must rise with current
        i = d.esw_vs_current[:, 0]
        assert d.switching_energy(i[-1]) > d.switching_energy(i[0])


def test_device_interpolation():
    dev = load_device(data_dir() / "devices" / "sj-mosfet-650v.yaml")
    assert abs(dev.rds_on(25) - 0.060) < 1e-12
    assert abs(dev.rds_on(50) - 0.069) < 1e-12          # midpoint of 25 and 75
    assert abs(dev.switching_energy(10) - 68e-6) < 1e-12
    assert abs(dev.switching_energy(10, Vds=200) - 34e-6) < 1e-12  # voltage scaling


def test_load_design_files():
    designs = sorted((data_dir() / "designs").glob("*.yaml"))
    assert len(designs) == 4
    kinds = set()
    devs = load_all_devices()
    for p in designs:
        d = load_design(p)
        kinds.add(d["kind"])
        assert d["device"] in devs
        assert d["Pout"] > 0 and d["fsw"] > 0
    assert kinds == {"buck", "boost", "pfc", "inverter"}


def test_load_design_rejects_bad_kind(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("name: x\nkind: flyback\ndevice: y\nPout: 1\nfsw: 1\n")
    with pytest.raises(ValueError):
        load_design(p)
    q = tmp_path / "missing.yaml"
    q.write_text("name: x\n")
    with pytest.raises(ValueError):
        load_design(q)


def test_device_loss_and_efficiency_curve():
    dev = load_device(data_dir() / "devices" / "syncbuck-fet-100v.yaml")
    loss = device_loss(dev, I_rms=10, I_sw=20, Vsw=48, fsw=200e3, Tj=100)
    assert abs(loss["conduction"] - 100 * 0.0034) < 1e-12
    assert abs(loss["switching"] - 19e-6 * 200e3) < 1e-9
    curve = efficiency_curve(dev, Vin=48, Vout=12, Pout_max=300, fsw=200e3, kind="buck")
    assert curve["efficiency"].shape == curve["Pout"].shape
    assert np.all(curve["efficiency"] > 0.9) and np.all(curve["efficiency"] < 1.0)
    # switching loss share falls with load, so efficiency rises then flattens
    assert curve["efficiency"][-1] > curve["efficiency"][0]
