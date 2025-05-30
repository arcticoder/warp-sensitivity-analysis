#!/usr/bin/env python3
"""
Generate Post-Newtonian corrections for warp drive metrics.
Takes baseline theory parameters and expands to specified PN order.
Outputs corrections as JSON-lines with AsciiMath metadata.
"""

import argparse
import json
import numpy as np
import sympy as sp
from sympy import symbols, expand, series, simplify
import sys
import os

# Add parent directory to path to use existing metadata parser
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze_sensitivity import parse_am_metadata

def compute_pn_expansion(v_over_c, pn_order=2):
    """
    Compute Post-Newtonian expansion for warp metric corrections.
    
    Args:
        v_over_c: Velocity parameter (v/c)
        pn_order: Maximum PN order to compute
    
    Returns:
        Dictionary of PN corrections by order
    """
    v = symbols('v', real=True, positive=True)
    c = symbols('c', real=True, positive=True)
    M = symbols('M', real=True, positive=True)  # Effective mass parameter
    r = symbols('r', real=True, positive=True)  # Characteristic scale
    
    # Baseline warp factor (simplified Alcubierre-like)
    # f(r) ≈ tanh(σ(r - R))
    # For PN expansion, we work with the metric perturbation h_μν
    
    # Leading order metric perturbation
    h_00_leading = M / (4 * sp.pi * r)  # Newtonian potential
    
    # PN corrections
    pn_corrections = {}
    
    for n in range(1, pn_order + 1):
        # nth PN correction ~ (v/c)^(2n)
        correction_factor = (v/c)**(2*n)
        
        if n == 1:
            # 1PN: velocity-dependent terms
            h_00_1pn = h_00_leading * correction_factor * (1 - (v/c)**2 / 2)
            h_ij_1pn = correction_factor * M / (4 * sp.pi * r) * sp.Matrix([
                [(v/c)**2, 0, 0],
                [0, (v/c)**2, 0],
                [0, 0, (v/c)**2]
            ])
            
            pn_corrections[f'{n}PN'] = {
                'h_00': str(simplify(h_00_1pn)),
                'h_spatial': str(h_ij_1pn),
                'order': 2*n,
                'physical_meaning': 'velocity_dependent_corrections'
            }
            
        elif n == 2:
            # 2PN: acceleration and self-energy terms  
            h_00_2pn = h_00_leading * correction_factor * (
                1 + (v/c)**4 / 8 - 3 * (M / (4 * sp.pi * r * c**2))**2
            )
            
            pn_corrections[f'{n}PN'] = {
                'h_00': str(simplify(h_00_2pn)),
                'h_spatial': 'higher_order_spatial_terms',
                'order': 2*n,
                'physical_meaning': 'acceleration_and_self_energy'
            }
        else:
            # Higher order placeholder
            h_00_npn = h_00_leading * correction_factor
            pn_corrections[f'{n}PN'] = {
                'h_00': str(simplify(h_00_npn)),
                'h_spatial': f'{n}pn_spatial_terms',
                'order': 2*n,
                'physical_meaning': f'order_{2*n}_corrections'
            }
    
    return pn_corrections

def estimate_observational_signatures(pn_corrections, frequency_range=(10, 1000)):
    """
    Estimate the observational signatures of PN corrections in GW detectors.
    """
    signatures = []
    
    for pn_order, correction in pn_corrections.items():
        # Rough estimate of strain amplitude scaling
        freq_low, freq_high = frequency_range
        frequencies = np.logspace(np.log10(freq_low), np.log10(freq_high), 50)
        
        # PN corrections typically scale as (πMf)^(2n/3) where n is PN order
        pn_number = correction['order'] // 2
        scaling_power = (2 * pn_number) / 3
        
        strain_scaling = (np.pi * 1e-6 * frequencies) ** scaling_power  # Rough scaling
        
        signatures.append({
            'pn_order': pn_order,
            'frequencies_hz': frequencies.tolist(),
            'strain_scaling': strain_scaling.tolist(),
            'peak_frequency': float(frequencies[np.argmax(strain_scaling)]),
            'scaling_power': scaling_power
        })
    
    return signatures

def main():
    parser = argparse.ArgumentParser(description="Compute Post-Newtonian corrections")
    parser.add_argument('--theory', required=True, help="Theory parameters (.am file)")
    parser.add_argument('--pn-config', required=True, help="PN configuration (.am file)")
    parser.add_argument('--out', required=True, help="Output PN corrections (.ndjson)")
    parser.add_argument('--oam', required=True, help="Output metadata (.am)")
    args = parser.parse_args()
    
    # Parse theory parameters
    warp_velocity = parse_am_metadata(args.theory, 'WarpVelocity') or '0.1'
    mass_parameter = parse_am_metadata(args.theory, 'MassParameter') or '1e-6'
    
    # Parse PN configuration
    max_pn_order = int(parse_am_metadata(args.pn_config, 'MaxPNOrder') or '2')
    frequency_range = parse_am_metadata(args.pn_config, 'FrequencyRange') or '[10, 1000]'
      # Parse frequency range
    if isinstance(frequency_range, str) and ',' in frequency_range:
        freq_range = [float(x.strip()) for x in frequency_range.split(',')]
    else:
        freq_range = [10, 1000]
    
    print(f"Computing PN corrections up to {max_pn_order}PN order...")
    
    # Compute PN corrections
    v_over_c = float(warp_velocity)
    pn_corrections = compute_pn_expansion(v_over_c, max_pn_order)
    
    # Estimate observational signatures
    signatures = estimate_observational_signatures(pn_corrections, freq_range)
    
    # Combine corrections with signatures
    results = []
    for i, (pn_order, correction) in enumerate(pn_corrections.items()):
        result = {
            'pn_order': pn_order,
            'correction': correction,
            'signature': signatures[i] if i < len(signatures) else None,
            'v_over_c': v_over_c,
            'mass_parameter': float(mass_parameter)
        }
        results.append(result)
    
    # Write JSON-lines output
    with open(args.out, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    # Write AsciiMath metadata
    summary = {
        'computation': 'post_newtonian_expansion',
        'max_pn_order': max_pn_order,
        'warp_velocity': warp_velocity,
        'mass_parameter': mass_parameter,
        'frequency_range': str(freq_range),
        'n_corrections': len(results)
    }
    
    with open(args.oam, 'w') as f:
        entries = ', '.join(f'{k} = {v!r}' for k, v in summary.items())
        f.write(f'[ {entries} ]\n')
    
    print(f"Wrote {args.out} ({len(results)} PN corrections) and {args.oam}")

if __name__ == '__main__':
    main()
