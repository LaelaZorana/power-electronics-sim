# Data files

## devices/

Four power device parameter files in YAML. Each holds Rds_on versus junction
temperature points, total switching energy Eon plus Eoff versus drain current
points at a stated reference voltage, gate and output charge figures, and the
junction to case and case to ambient thermal resistances.

The numbers are representative of their datasheet class and are not copied
from any specific manufacturer part. They sit at typical values for the class
so that loss and efficiency calculations come out realistic, but they should
not be used to qualify hardware.

| File | Class |
|---|---|
| sj-mosfet-650v.yaml | 650 V silicon superjunction MOSFET, 60 mohm class |
| sic-mosfet-650v.yaml | 650 V SiC MOSFET, 50 mohm class |
| syncbuck-fet-100v.yaml | 100 V synchronous buck FET, 2.4 mohm class |
| sic-halfbridge-1200v.yaml | 1200 V SiC half bridge module, per switch values |

`pesim.devices.load_device` reads one file, `load_all_devices` reads the
directory, and `DeviceData.rds_on(Tj)` and `DeviceData.switching_energy(I, V)`
interpolate the tables. Switching energy scales linearly with the switched
voltage relative to the tabulated reference.

## designs/

Four converter design points in YAML, each naming one of the devices above.

| File | Design |
|---|---|
| buck-48to12-300w.yaml | 48 V to 12 V 300 W intermediate bus buck |
| boost-pv-400v.yaml | 250 V PV string to 400 V boost, 2 kW |
| pfc-3kw-frontend.yaml | 3 kW totem pole PFC style front end, run through the averaged boost PFC model |
| inverter-ev-10kw.yaml | 10 kW EV traction inverter operating point, 800 V bus |

`pesim.devices.load_design` reads one file. `examples/design_comparison.py`
loads all four, sizes and designs each converter, simulates it, and writes a
comparison table of efficiency, ripple and loop margins to
`examples/design_comparison.md`.
