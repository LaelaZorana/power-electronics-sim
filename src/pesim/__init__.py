"""pesim: switching power converter simulation and design library."""
from .converters import Converter, ConverterSpec, simulate, SimResult, ripple_formulas
from .design import (size_inductor, size_capacitor, ccm_boundary_load,
                     ccm_boundary_inductance, buck_small_signal, boost_small_signal,
                     buckboost_small_signal, control_to_output, boost_rhp_zero,
                     design_pi, design_type2, design_type3, loop_metrics, phase_deg)
from .inverter import (spwm, svpwm, line_line, thd, spectrum, fundamental,
                       dc_utilisation, dead_time_voltage_error)
from .pfc import boost_pfc
from .thermal import MosfetParams, mosfet_losses, junction_temperature, rds_on_at
from .devices import (DeviceData, load_device, load_design, load_all_devices,
                      device_loss, efficiency_curve)

__all__ = [
    "Converter", "ConverterSpec", "simulate", "SimResult", "ripple_formulas",
    "size_inductor", "size_capacitor", "ccm_boundary_load", "ccm_boundary_inductance",
    "buck_small_signal", "boost_small_signal", "buckboost_small_signal",
    "control_to_output", "boost_rhp_zero",
    "design_pi", "design_type2", "design_type3", "loop_metrics", "phase_deg",
    "spwm", "svpwm", "line_line", "thd", "spectrum", "fundamental",
    "dc_utilisation", "dead_time_voltage_error",
    "boost_pfc",
    "MosfetParams", "mosfet_losses", "junction_temperature", "rds_on_at",
    "DeviceData", "load_device", "load_design", "load_all_devices",
    "device_loss", "efficiency_curve",
]
