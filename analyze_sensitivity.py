#!/usr/bin/env python3
"""
Compare synthetic signals against a detector noise curve:
 - For each mock signal (from NDJSON + AsciiMath metadata),
   compute its FFT amplitude spectrum.
 - Overlay on the noise PSD (from CSV + AsciiMath metadata).
 - Flag signals whose peak amplitude exceeds the noise floor (SNR > 1).
 - Write out a JSON‐lines summary and an AsciiMath summary.
"""

import argparse
import csv
import json
import re

import numpy as np
import ndjson
from scipy.fft import rfft, rfftfreq

def load_mock_signals(ndjson_path):
    with open(ndjson_path) as f:
        return ndjson.load(f)

def load_noise_curve(csv_path):
    freqs = []
    noise = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            f_hz, n_val = map(float, row)
            freqs.append(f_hz)
            noise.append(n_val)
    return np.array(freqs), np.array(noise)

def parse_am_metadata(am_path, key):
    """
    Simple AsciiMath metadata parser: looks for key = "value" or key = value.
    """
    txt = open(am_path).read()
    # Remove brackets
    txt = txt.strip().lstrip('[').rstrip(']')
    parts = [p.strip() for p in txt.split(',')]
    for part in parts:
        if part.startswith(key):
            # split on '=' and strip quotes
            _, val = part.split('=', 1)
            val = val.strip().strip('"').strip()
            return val
    return None

def main():
    p = argparse.ArgumentParser(description="Analyze signal detectability vs. noise curve")
    p.add_argument('--mock',   required=True, help="mock_data.ndjson")
    p.add_argument('--meta',   required=True, help="mock_data.am")
    p.add_argument('--noise',  required=True, help="sensitivity_curve.csv")
    p.add_argument('--nmeta',  required=True, help="sensitivity_curve.am")
    p.add_argument('--out',    required=True, help="sensitivity_comparison.ndjson")
    p.add_argument('--oam',    required=True, help="sensitivity_comparison.am")
    p.add_argument('--threshold', type=float, default=1.0,
                   help="SNR threshold for detectability (default: 1.0)")
    args = p.parse_args()

    # Load inputs
    signals = load_mock_signals(args.mock)
    noise_freq, noise_psd = load_noise_curve(args.noise)    # Parse metadata
    detector = parse_am_metadata(args.nmeta, 'Model') or 'unknown_detector'
    
    # Parse mock data metadata
    mock_sampling_rate = parse_am_metadata(args.meta, 'SamplingRate')
    mock_noise_model = parse_am_metadata(args.meta, 'NoiseModel')
    mock_injection_count = parse_am_metadata(args.meta, 'InjectionCount')

    results = []
    for sig in signals:
        label = sig.get('label', 'unknown')
        ts = np.array(sig['time_series'], dtype=float)
        fs = float(sig.get('sampling_rate', 1.0))

        # FFT
        yf = rfft(ts)
        amp = np.abs(yf) / len(ts) * 2  # single-sided amplitude spectrum
        xf = rfftfreq(len(ts), 1/fs)

        # Interpolate noise PSD onto signal freq grid
        noise_interp = np.interp(xf, noise_freq, noise_psd)
        snr = amp / noise_interp
        max_snr = float(np.max(snr))

        detectable = max_snr > args.threshold

        results.append({
            'label':       label,
            'detectable':  detectable,
            'snr':         max_snr        })

    # Write JSON-lines output
    with open(args.out, 'w') as outf:
        for result in results:
            outf.write(json.dumps(result) + '\n')

    # Write AsciiMath summary
    summary = {
        'detector':     detector,
        'n_signals':    len(results),
        'threshold':    args.threshold
    }
    
    # Add mock metadata to summary if available
    if mock_sampling_rate:
        summary['sampling_rate'] = mock_sampling_rate
    if mock_noise_model:
        summary['noise_model'] = mock_noise_model
    if mock_injection_count:
        summary['injection_count'] = mock_injection_count
    with open(args.oam, 'w') as amf:
        entries = ', '.join(f'{k} = {v!r}' for k, v in summary.items())
        amf.write(f'[ {entries} ]\n')

    print(f"Wrote {args.out} ({len(results)} records) and {args.oam}")

if __name__ == '__main__':
    main()