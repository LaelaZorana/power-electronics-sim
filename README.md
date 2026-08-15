# pesim: switching power converter simulation and design

A buck, a boost, a buck-boost, a three phase inverter, a boost PFC stage and a
MOSFET loss and thermal budget, all simulated in plain numpy and scipy with
python-control handling the loop design. The validation table below lists the
numbers that are asserted by the test suite and names the test for each one.
The paragraph after it lists the numbers that come from running
`examples/run_all.py`, and each of those is also pinned by a test.

## Theory summary

Converters live in `pesim.converters`. Buck, boost and buck-boost are switched
state space systems with state x = [iL, vC], and each converter has three
affine sub circuits: switch on, diode conducting, and inductor idle for DCM.
Each sub circuit is discretised exactly with a matrix exponential of the
augmented [A b; 0 0] matrix, so the fixed time step only sets how finely the
switching instants and the DCM boundary are resolved.

Loss elements are MOSFET Rds_on, inductor DCR, diode forward drop and
capacitor ESR, and the ESR also produces the correct ripple step. CCM or DCM
is detected from whether the idle mode is ever entered in the last periods.
Periodic steady state is found by Newton shooting on the one period map, which
means simulations start settled and a few periods are enough. Efficiency is
mean output power over mean input power, per element losses are reported, and
the power balance Pin equals Pout plus the sum of losses closes to better than
0.1 percent, which `test_power_balance` asserts.

The duty cycle is realised as an integer number of simulation steps. The
effective value is stored on the result as `D_eff` and the simulator warns
when quantisation moves it by more than 1e-3. Specs are validated on
construction, a duty cycle outside [0, 1) or a nonpositive component value
raises immediately. The buck-boost output is reported as a positive magnitude,
the physical output of the inverting topology is negative.

Design lives in `pesim.design`. It covers inductor and capacitor sizing from
ripple ratios, the CCM boundary load and critical inductance, and averaged CCM
small signal control to output models:

- buck: Gvd = Vin / (1 + sL/R + s^2 LC)
- boost: Gvd = (Vo/D') (1 - sL/(D'^2 R)) / (1 + sL/(D'^2 R) + s^2 LC/D'^2), right half plane zero at D'^2 R / L
- buck-boost: same denominator, zero at D'^2 R / (D L)

An optional ESR zero of 1 plus s ESR C multiplies each. Compensators come from
the k-factor method: type II with one zero, one pole and an integrator, type
III with two of each, and a plain PI. Each is scaled for unity loop gain at
the requested crossover, and phase margin and gain margin are read back from
`control.margin`. Every design function measures the loop it hands back. It
raises ValueError when the requested phase boost is beyond what the
compensator type can supply or when the measured loop is unstable, and it
warns when the loop has more than one gain crossover or misses the requested
crossover frequency, so a design that cannot work never comes back silently.
`test_design_raises_on_unachievable_or_unstable` exercises the failure paths.

The inverter in `pesim.inverter` is a three phase two level VSI with pole
voltages at plus or minus Vdc/2. SPWM compares sine references with a
triangular carrier, and SVPWM is implemented as SPWM with min max zero
sequence injection, which with a triangular carrier and the resulting equal
zero vector split is equivalent to symmetric space vector modulation and
extends the linear range from m = 1 to m = 2/sqrt(3). Line to line voltages,
FFT harmonic spectra and THD are provided. The THD here counts everything
above the fundamental including the PWM carrier sidebands, so on a raw PWM
voltage it is the unfiltered voltage THD. Dead time is not switched in the
model. The module docstring explains its effect, a current sign dependent
average voltage error of magnitude Vdc td fsw plus low order odd harmonics,
and `dead_time_voltage_error()` gives the magnitude.

The PFC in `pesim.pfc` is an averaged boost with an inner average current PI
loop, feed forward duty plus PI, tracking k times the rectified input voltage,
and a slow outer voltage loop scaling k. It reports input current THD, power
factor and the twice line frequency output ripple. The residual THD at the
default settings is dominated by the 3rd harmonic that the outer voltage
loop's proportional gain injects by multiplying the output ripple into the
current reference. Setting that gain to zero drops the THD below 0.5 percent
with power factor 1.000 while tripling the current loop gain changes nothing,
and `test_pfc_thd_mechanism_is_outer_loop_ripple` pins that experiment.

Thermal in `pesim.thermal` builds a MOSFET loss budget from conduction loss
with a linear Rds_on temperature coefficient, switching loss from a Miller
charge transition time estimate, Coss and reverse recovery losses, and gate
drive loss. Because the total loss is affine in Tj the self consistent
junction temperature has a closed form, and when the thermal feedback gain
reaches one there is no finite operating point and the solver raises a
thermal runaway error instead of returning a divergent number, which
`test_thermal_runaway_raises` checks.

## Device data and designs

`data/devices/` holds parameter files for four power devices, a 650 V
superjunction MOSFET, a 650 V SiC MOSFET, a 100 V synchronous buck FET and a
1200 V SiC half bridge module, each with Rds_on versus temperature points,
switching energy versus current points and thermal resistances. The values
are representative of their datasheet class, not copies of a specific part,
see `data/README.md`. `data/designs/` holds four converter design points, a
48 V to 12 V 300 W buck, a 400 V PV boost, a 3 kW PFC front end and a 10 kW
EV traction inverter operating point. `pesim.devices` loads both kinds of
file and interpolates the tables, and `examples/design_comparison.py` designs,
simulates and compares all four, writing a table of efficiency, ripple and
loop margins to `examples/design_comparison.md`. Efficiency curves there
combine conduction loss from the interpolated Rds_on with switching loss from
the tabulated energies, so they fall at light load where switching dominates
and at heavy load where conduction grows.

## API

```python
from pesim import ConverterSpec, simulate
spec = ConverterSpec("buck", Vin=12, D=0.4, fs=100e3, L=100e-6, C=100e-6, R=5, Rds_on=0.02)
r = simulate(spec)               # SimResult: t, iL, vout, mode, D_eff, Vout_avg, dv_pp, di_pp, efficiency, losses
from pesim.design import buck_small_signal, design_type3, ccm_boundary_load, size_inductor
G = buck_small_signal(spec)
Gc, metrics = design_type3(G, fc=5e3, pm_target=60)
from pesim.inverter import spwm, svpwm, line_line, thd, spectrum
w = spwm(400, 0.8, f1=50, fsw=5e3); print(thd(line_line(w), w["t"], 50))
from pesim.pfc import boost_pfc
from pesim.thermal import MosfetParams, junction_temperature
from pesim.devices import load_all_devices, load_design, efficiency_curve
```

## Validation table

| Check | Closed form | Simulation | Test |
|---|---|---|---|
| Buck Vout, D = 0.4, 12 V | 4.800 V | 4.800 V | test_buck_steady_state_vout |
| Boost Vout, D = 0.5, 12 V | 24.00 V | 24.00 V | test_boost_steady_state_vout |
| Buck-boost Vout, D = 0.4, 12 V | 8.000 V | within 1 % | test_buckboost_steady_state_and_ripple |
| Buck ripple di, dv | 0.288 A, 3.6 mV | within 3 % and 5 % | test_buck_ripple_formulas |
| Boost ripple di, dv | 0.600 A, 60 mV | within 3 % and 5 % | test_boost_ripple_formulas |
| Buck CCM boundary R = 2L/((1-D)T) | 33.3 ohm | CCM at 0.8 Rb, DCM at 1.25 Rb | test_ccm_boundary |
| Buck-boost CCM boundary R = 2L/((1-D)^2 T) | 55.6 ohm | CCM at 0.9 Rb, DCM at 1.1 Rb | test_buckboost_ccm_boundary |
| Boost DCM voltage M = (1+sqrt(1+4D^2/K))/2 | 25.90 V at R = 200 | within 0.1 % | test_boost_dcm_closed_form |
| Power balance Pin = Pout + losses | exact | closes to 0.1 % | test_power_balance |
| Boost RHP zero D'^2 R/L | 50 krad/s, 7.96 kHz | tf zero matches to 1e-9 | test_boost_rhp_zero |
| Type III at 5 kHz, 60 deg, ESR plant | target | 5000 Hz, 60.0 deg, GM 33.5 dB | test_type3_hits_targets |
| SPWM fundamental | m Vdc/2 | within 2 % for m = 0.4, 0.8, 1.0 | test_spwm_fundamental |
| SVPWM linear to m = 1.155 | m Vdc/2 | within 2 %, SPWM saturated | test_svpwm_extends_linear_range |
| Utilisation linear limits | 0.866 and 1.0 Vdc | within 2 % | test_utilisation_linear_limits |
| Square wave THD | sqrt(pi^2/8 - 1) = 48.34 % | matches to 0.5 % | test_square_wave_thd |
| MOSFET loss mechanisms | hand computed W | exact | test_mosfet_losses_pinned |
| Thermal closed form Tj | 48.069 C by hand | exact | test_thermal_self_consistent_pinned |

Other numbers printed by `examples/run_all.py`, each pinned by the named
test: the buck in DCM at 2.5 times the boundary load rises to 6.58 V,
test_buck_dcm_example_voltage. The SPWM and SVPWM linear ranges end at 0.866
and 1.0 Vdc of line to line fundamental, test_utilisation_linear_limits, and
the utilisation figure sweeps m only to the SVPWM linear limit of 1.155. The
boost PFC pulls input current at 4.5 percent THD with PF 0.998,
test_pfc_baseline_metrics. The example MOSFET at 100 kHz runs at Tj of 47.2 C,
test_thermal_example_tj. The boost efficiency figure combines the switched
simulation's conduction losses with switching loss from `mosfet_losses`, so
the curve peaks near 22 W at 96 percent and falls toward both light load,
where switching loss dominates, and heavy load, where conduction grows, and
its power axis comes from the simulated output voltage rather than the ideal
one.

## Figures

`figures/buck_ccm_vs_dcm.png`, `figures/boost_efficiency_vs_load.png`,
`figures/control_loop_bode.png`, `figures/inverter_waveforms_fft.png`,
`figures/svpwm_vs_spwm_utilisation.png`, `figures/pfc_input_current.png`,
`figures/mosfet_thermal.png`.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
python -m pytest -q
python examples/run_all.py            # writes figures/ and prints key numbers
python examples/design_comparison.py  # writes examples/design_comparison.md
```

The simulators are plain Python loops, exact per step but not vectorised. The
whole suite runs in about one second.

## License

MIT, see LICENSE.
