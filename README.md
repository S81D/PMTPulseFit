# PMTPulseFit

### Usage
```python3 pulse_fitting.py```

### Description
This code gathers raw ADC traces and their corresponding hits information to collect a sample of SPE pulses. These SPE pulses are then fit with a 7 parameter function taken from Daya Bay to properly describe the PMT response seen in data. These fits are then used by the PMTWaveformSim tool to reconstruct MC pulses using the WCSim true photon information.

Script is well commented. Please adjust accordingly.