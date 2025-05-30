#!/usr/bin/env python3
"""
Produce a toy detector noise curve (frequency vs. strain-spectral-density)
and write out both a CSV and a small .am metadata summary.
"""

import numpy as np
import csv
import ast

# ——— Define a toy PSD shape ———
def toy_psd(f):
    # simple power-law plus white floor:
    # S_n(f) = S0 * ( (f0 / f)**4 + 1 )
    S0 = 1e-23    # baseline strain /√Hz
    f0 = 100.0    # knee frequency [Hz]
    return S0 * ( (f0 / f)**4 + 1 )

# ——— Generate the curve ———
frequencies = np.logspace(1, 4, 200)  # 10 Hz → 10 kHz, 200 points
noise     = toy_psd(frequencies)

# ——— Write CSV ———
with open('sensitivity_curve.csv', 'w', newline='') as csvf:
    writer = csv.writer(csvf)
    writer.writerow(['frequency_Hz','noise_strain_per_sqrtHz'])
    for f, n in zip(frequencies, noise):
        writer.writerow([f, n])

# ——— Write simple AsciiMath metadata ———
with open('sensitivity_curve.am', 'w') as amf:
    amf.write('[ ')
    amf.write('Model = "toy_powerlaw", ')
    amf.write('S0 = 1e-23, ')
    amf.write('f0 = 100, ')
    amf.write('n_points = 200 ')
    amf.write(']\n')

print("Wrote sensitivity_curve.csv and sensitivity_curve.am")