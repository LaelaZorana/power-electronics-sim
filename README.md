# pesim: switching power converter simulation and design

A small Python library for simulating and designing switching converters, three-phase inverters, a boost PFC stage and MOSFET thermal budgets. Everything is plain numpy and scipy plus python-control for the loop design. All results below are produced by `examples/run_all.py` and checked by `tests/`.

## Theory summary

Converters (`pesim.converters`). Buck, boost and buck-boost are modelled as switched state-space systems with state x = [iL, vC]. Each converter has three affine sub-circuits (switch on, diode conducting, inductor idle for DCM). Each sub-circuit is discretised exactly with a matrix exponential of the augmented [A b; 0 0] matrix, so the fixed time step only sets how finely the switching instants and DCM boundary are resolved. Loss elements are MOSFET Rds_on, inductor DCR, diode forward drop and capacitor ESR (which also gives the correct ESR ripple step). CCM or DCM is detected from whether the idle mode is ever entered in the last periods. Periodic steady state is found by Newton shooting on the one-period map, so simulations start settled and a few periods are enough. Efficiency is mean output power over mean input power, and per-element losses are reported.

Design (`pesim.design`). Inductor and capacitor sizing from ripple ratios, CCM boundary load and critical inductance, and averaged CCM small-signal control-to-output models:

- buck: Gvd = Vin / (1 + sL/R + s^2 LC)
- boost: Gvd = (Vo/D') (1 - sL/(D'^2 R)) / (1 + sL/(D'^2 R) + s^2 LC/D'^2), right-half-plane zero at D'^2 R / L
- buck-boost: same denominator, zero at D'^2 R / (D L)

An optional ESR zero (1 + s ESR C) multiplies each. Compensators are designed with the k-factor method: type II (one zero, one pole, integrator), type III (two zeros, two poles, integrator) and a plain PI, each scaled for unity loop gain at the requested crossover, with phase margin and gain margin read back from `control.margin`.

Inverter (`pesim.inverter`). Three-phase two-level VSI with pole voltages at plus or minus Vdc/2. SPWM compares sine references with a triangular carrier. SVPWM is implemented as SPWM with min-max zero-sequence injection, which is equivalent to symmetric space vector modulation and extends the linear range from m = 1 to m = 2/sqrt(3). Line-to-line voltages, FFT harmonic spectra and THD are provided. Dead time is not switched in the model; the module docstring explains its effect (a current-sign-dependent voltage error of Vdc td fsw per period, low-order odd harmonics) and `dead_time_voltage_error()` gives the magnitude.

PFC (`pesim.pfc`). Averaged boost PFC with an inner average-current PI loop (feed-forward duty plus PI) tracking k times the rectified input voltage, and a slow outer voltage loop scaling k. Reports input current THD, power factor and the twice-line-frequency output ripple.

Thermal (`pesim.thermal`). MOSFET conduction loss from Rds_on with a linear temperature coefficient, switching loss from a Miller-charge transition-time estimate, Coss and reverse-recovery losses, gate drive loss, and a self-consistent junction temperature through Rth_jc plus Rth_ca.

## API

```python
from pesim import ConverterSpec, simulate
spec = ConverterSpec("buck", Vin=12, D=0.4, fs=100e3, L=100e-6, C=100e-6, R=5, Rds_on=0.02)
r = simulate(spec)               # SimResult: t, iL, vout, mode, Vout_avg, dv_pp, di_pp, efficiency, losses
from pesim.design import buck_small_signal, design_type3, ccm_boundary_load, size_inductor
G = buck_small_signal(spec)
Gc, metrics = design_type3(G, fc=5e3, pm_target=60)
from pesim.inverter import spwm, svpwm, line_line, thd, spectrum
w = spwm(400, 0.8, f1=50, fsw=5e3); print(thd(line_line(w), w["t"], 50))
from pesim.pfc import boost_pfc
from pesim.thermal import MosfetParams, junction_temperature
```

## Validation table

| Check | Closed form | Simulation | Test |
|---|---|---|---|
| Buck Vout, D = 0.4, 12 V | 4.800 V | 4.800 V | test_buck_steady_state_vout |
| Boost Vout, D = 0.5, 12 V | 24.00 V | 24.00 V | test_boost_steady_state_vout |
| Buck ripple di, dv | 0.288 A, 3.6 mV | within 3 % and 5 % | test_buck_ripple_formulas |
| Boost ripple di, dv | 0.600 A, 60 mV | within 3 % and 5 % | test_boost_ripple_formulas |
| Buck CCM boundary R = 2L/((1-D)T) | 33.3 ohm | CCM at 0.8 Rb, DCM at 1.25 Rb | test_ccm_boundary |
| Boost RHP zero D'^2 R/L | 50 krad/s (7.96 kHz) | tf zero matches to 1e-9 | test_boost_rhp_zero |
| Type III at 5 kHz, 60 deg | target | 5000 Hz, 60.0 deg, GM 33.5 dB | test_type3_hits_targets |
| SPWM fundamental | m Vdc/2 | within 2 % for m = 0.4, 0.8, 1.0 | test_spwm_fundamental |
| SVPWM linear to m = 1.155 | m Vdc/2 | within 2 %, SPWM saturated | test_svpwm_extends_linear_range |
| Square wave THD | sqrt(pi^2/8 - 1) = 48.34 % | matches to 0.5 % | test_square_wave_thd |
| Thermal fixed point | Tj = Ta + P Rth | converged | test_thermal_converges |

Other numbers from `examples/run_all.py`: buck in DCM at 2.5 times the boundary load rises to 6.58 V; lossy boost peaks at 97.9 % efficiency; SPWM line-line utilisation saturates at 0.866 Vdc while SVPWM reaches 1.0 Vdc in the linear range; boost PFC input THD 4.5 % with PF 0.998; example MOSFET at 100 kHz runs at Tj about 47 C.

## Figures

`figures/buck_ccm_vs_dcm.png`, `figures/boost_efficiency_vs_load.png`, `figures/control_loop_bode.png`, `figures/inverter_waveforms_fft.png`, `figures/svpwm_vs_spwm_utilisation.png`, `figures/pfc_input_current.png`, `figures/mosfet_thermal.png`.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy matplotlib control pytest
python -m pytest -q          # pytest.ini adds src/ to the path
python examples/run_all.py   # writes figures/ and prints key numbers
```
