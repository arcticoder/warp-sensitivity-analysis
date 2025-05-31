#!/usr/bin/env python3
"""
Analyze Post-Newtonian corrections against experimental precision tests.
Computes goodness-of-fit, residuals, and detection prospects for various PN orders.
"""

import argparse
import csv
import json
import numpy as np
import ndjson
import sys
import os
from scipy import stats, interpolate

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze_sensitivity import parse_am_metadata

def load_experimental_data(csv_path):
    """Load experimental sensitivity data from CSV."""
    frequencies = []
    sensitivities = []
    exp_types = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frequencies.append(float(row['frequency_hz']))
            sensitivities.append(float(row['strain_sensitivity']))
            exp_types.append(row['experiment_type'])
    
    return np.array(frequencies), np.array(sensitivities), exp_types

def load_pn_corrections(ndjson_path):
    """Load PN corrections from JSON-lines file."""
    with open(ndjson_path, 'r') as f:
        return ndjson.load(f)

def compute_chi_squared(predicted_strain, observed_sensitivity, uncertainties=None):
    """Compute chi-squared goodness of fit."""
    if uncertainties is None:
        uncertainties = observed_sensitivity * 0.1  # 10% uncertainty assumption
    
    chi_sq = np.sum(((predicted_strain - observed_sensitivity) / uncertainties) ** 2)
    dof = len(predicted_strain) - 1  # degrees of freedom
    p_value = 1 - stats.chi2.cdf(chi_sq, dof)
    
    return chi_sq, dof, p_value

def estimate_detection_snr(pn_correction, frequencies, experimental_sensitivity):
    """Estimate detection SNR for a given PN correction."""
    if 'signature' not in pn_correction or pn_correction['signature'] is None:
        return 0.0
    
    sig_freqs = np.array(pn_correction['signature']['frequencies_hz'])
    sig_strain = np.array(pn_correction['signature']['strain_scaling'])
    
    # Normalize strain scaling to physical units (rough estimate)
    sig_strain *= pn_correction['mass_parameter'] * 1e-21  # Scale to typical GW strain
    
    # Interpolate onto experimental frequency grid
    if len(sig_freqs) > 1 and len(frequencies) > 1:
        # Find overlapping frequency range
        f_min = max(np.min(sig_freqs), np.min(frequencies))
        f_max = min(np.max(sig_freqs), np.max(frequencies))
        
        if f_min < f_max:
            f_overlap = frequencies[(frequencies >= f_min) & (frequencies <= f_max)]
            if len(f_overlap) > 0:
                interp_func = interpolate.interp1d(sig_freqs, sig_strain, 
                                                 bounds_error=False, fill_value=0)
                predicted_strain = interp_func(f_overlap)
                
                exp_sens_overlap = experimental_sensitivity[(frequencies >= f_min) & (frequencies <= f_max)]
                
                # Compute SNR
                snr_squared = np.sum((predicted_strain / exp_sens_overlap) ** 2)
                return np.sqrt(snr_squared)
    
    return 0.0

def analyze_pn_order(pn_correction, exp_frequencies, exp_sensitivity, exp_types):
    """Analyze a specific PN order against experimental data."""
    results = {}
    
    # Group experimental data by type
    unique_exp_types = list(set(exp_types))
    
    for exp_type in unique_exp_types:
        exp_mask = np.array(exp_types) == exp_type
        freq_subset = exp_frequencies[exp_mask]
        sens_subset = exp_sensitivity[exp_mask]
        
        if len(freq_subset) == 0:
            continue
            
        # Compute detection SNR
        snr = estimate_detection_snr(pn_correction, freq_subset, sens_subset)
          # Estimate constraints on theory parameters
        if snr > 0:
            # Very rough constraint estimate
            constraint_strength = snr / 5.0  # SNR=5 gives order-of-magnitude constraint
            param_bound = pn_correction['mass_parameter'] * (1.0 / constraint_strength)
        else:
            param_bound = None
        
        results[exp_type] = {
            'snr': float(snr),
            'detectable': bool(snr > 5.0),  # 5-sigma detection threshold
            'parameter_bound': param_bound,
            'frequency_range': [float(np.min(freq_subset)), float(np.max(freq_subset))],
            'n_data_points': int(len(freq_subset))
        }
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Analyze PN corrections vs experimental data")
    parser.add_argument('--pn-data', required=True, help="PN corrections (.ndjson)")
    parser.add_argument('--pn-meta', required=True, help="PN metadata (.am)")
    parser.add_argument('--exp-data', required=True, help="Experimental data (.csv)")
    parser.add_argument('--exp-meta', required=True, help="Experimental metadata (.am)")
    parser.add_argument('--out', required=True, help="Analysis results (.ndjson)")
    parser.add_argument('--oam', required=True, help="Analysis summary (.am)")
    parser.add_argument('--snr-threshold', type=float, default=5.0,
                       help="SNR threshold for detectability")
    args = parser.parse_args()
    
    # Load data
    pn_corrections = load_pn_corrections(args.pn_data)
    exp_frequencies, exp_sensitivity, exp_types = load_experimental_data(args.exp_data)
    
    # Parse metadata
    max_pn_order = parse_am_metadata(args.pn_meta, 'max_pn_order')
    warp_velocity = parse_am_metadata(args.pn_meta, 'warp_velocity')
    exp_type = parse_am_metadata(args.exp_meta, 'ExperimentType')
    
    print(f"Analyzing {len(pn_corrections)} PN corrections against {len(exp_frequencies)} experimental points...")
      # Analyze each PN order
    results = []
    for pn_correction in pn_corrections:
        analysis = analyze_pn_order(pn_correction, exp_frequencies, exp_sensitivity, exp_types)
        
        result = {
            'pn_order': pn_correction['pn_order'],
            'theory_params': {
                'v_over_c': pn_correction['v_over_c'],
                'mass_parameter': pn_correction['mass_parameter']
            },
            'experimental_analysis': analysis,
            'overall_detectability': bool(any(exp['detectable'] for exp in analysis.values())),
            'best_snr': float(max((exp.get('snr', 0) for exp in analysis.values()), default=0.0))
        }
        results.append(result)
    
    # Write results
    with open(args.out, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    # Summary statistics
    detectable_orders = [r for r in results if r['overall_detectability']]
    best_constraint = min((r['theory_params']['mass_parameter'] for r in detectable_orders),
                         default=None)
    
    summary = {
        'analysis_type': 'pn_experimental_comparison',
        'n_pn_orders': len(results),
        'n_detectable': len(detectable_orders),
        'max_pn_order_analyzed': max_pn_order,
        'warp_velocity': warp_velocity,
        'snr_threshold': args.snr_threshold,
        'best_constraint': best_constraint,
        'experiment_types': list(set(exp_types))
    }
    
    with open(args.oam, 'w') as f:
        entries = ', '.join(f'{k} = {v!r}' for k, v in summary.items())
        f.write(f'[ {entries} ]\n')
    
    print(f"Wrote {args.out} ({len(results)} analyses) and {args.oam}")
    print(f"Found {len(detectable_orders)} potentially detectable PN orders")

if __name__ == '__main__':
    main()
