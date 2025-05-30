#!/usr/bin/env python3
"""
Generate toy 2D black hole models for strong curvature regime analysis.
Computes curvature invariants and Planck-scale effects.
"""

import argparse
import json
import numpy as np
import sympy as sp
from sympy import symbols, Matrix, simplify, diff, sqrt, exp, tanh
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze_sensitivity import parse_am_metadata

def compute_2d_schwarzschild_curvature(mass_param, planck_length):
    """
    Compute curvature invariants for 2D Schwarzschild-like metric.
    
    Metric: ds² = -(1 - 2M/r) dt² + (1 - 2M/r)⁻¹ dr²
    """
    r, t, M, l_p = symbols('r t M l_p', real=True, positive=True)
    
    # Metric components
    f = 1 - 2*M/r
    g_tt = -f
    g_rr = 1/f
    
    # Simple curvature invariants for 2D case
    # Ricci scalar R ≈ 2M/r³ for large r
    ricci_scalar = 2*M/r**3
    
    # Kretschmann scalar K = R_{μνρσ} R^{μνρσ}
    # For Schwarzschild: K = 48M²/r⁶
    kretschmann = 48*M**2/r**6
    
    # Planck-scale corrections
    # When curvature ~ 1/l_p², quantum effects dominate
    planck_curvature = 1/l_p**2
    quantum_parameter = kretschmann * l_p**4  # Dimensionless quantum strength
    
    # Regions of validity
    classical_regime = kretschmann * l_p**4 < 1  # Classical GR valid
    quantum_regime = kretschmann * l_p**4 >= 1   # Quantum corrections important
    
    return {
        'metric_type': '2d_schwarzschild',
        'ricci_scalar': str(ricci_scalar),
        'kretschmann_scalar': str(kretschmann),
        'planck_curvature': str(planck_curvature),
        'quantum_parameter': str(quantum_parameter),
        'classical_regime_condition': str(classical_regime),
        'quantum_regime_condition': str(quantum_regime),
        'mass_parameter': float(mass_param),
        'planck_length': float(planck_length)
    }

def compute_warp_bubble_curvature(warp_factor, bubble_thickness, planck_length):
    """
    Compute curvature for warp bubble in strong-field regime.
    """
    r, R, sigma, l_p = symbols('r R sigma l_p', real=True, positive=True)
    
    # Warp factor f(r) = tanh(σ(r - R))
    f = tanh(sigma * (r - R))
    
    # Approximate curvature at bubble wall
    # Second derivative gives curvature scale
    f_prime = diff(f, r)
    f_double_prime = diff(f_prime, r)
    
    # Curvature scale ~ σ²
    curvature_scale = sigma**2
    
    # When σ ~ 1/l_p, we hit Planck scale
    planck_sigma = 1/l_p
    
    # Quantum parameter
    quantum_param = (sigma * l_p)**2
    
    return {
        'metric_type': 'warp_bubble',
        'warp_factor': str(f),
        'curvature_scale': str(curvature_scale),
        'planck_sigma': str(planck_sigma),
        'quantum_parameter': str(quantum_param),
        'bubble_thickness': float(bubble_thickness),
        'planck_length': float(planck_length)
    }

def analyze_geodesics(model_data, initial_conditions):
    """
    Simple geodesic analysis for toy model.
    """
    if model_data['metric_type'] == '2d_schwarzschild':
        # Effective potential analysis
        r_values = np.logspace(-2, 2, 100)  # From 0.01 to 100 in natural units
        M = model_data['mass_parameter']
        
        # Effective potential V_eff = -(1 - 2M/r) for radial geodesics
        V_eff = -(1 - 2*M/r_values)
        
        # Find turning points, event horizon, etc.
        event_horizon = 2*M
        photon_sphere = 3*M  # Not applicable in 2D, but keep for consistency
        
        return {
            'geodesic_type': 'radial_timelike',
            'r_values': r_values.tolist(),
            'effective_potential': V_eff.tolist(),
            'event_horizon': event_horizon,
            'photon_sphere': photon_sphere,
            'classical_valid': True
        }
    
    elif model_data['metric_type'] == 'warp_bubble':
        # Simple trajectory through warp bubble
        r_values = np.linspace(-5, 5, 100)
        bubble_center = 0
        thickness = model_data['bubble_thickness']
        
        # Warp factor profile
        warp_profile = np.tanh(r_values / thickness)
        
        return {
            'geodesic_type': 'through_bubble',
            'r_values': r_values.tolist(),
            'warp_profile': warp_profile.tolist(),
            'bubble_center': bubble_center,
            'thickness': thickness,
            'classical_valid': np.all(np.abs(warp_profile) < 0.9)  # Avoid extreme warping
        }
    
    return {}

def main():
    parser = argparse.ArgumentParser(description="Generate 2D black hole toy models")
    parser.add_argument('--model-config', required=True, help="Model configuration (.am)")
    parser.add_argument('--out', required=True, help="Model results (.ndjson)")
    parser.add_argument('--oam', required=True, help="Model metadata (.am)")
    args = parser.parse_args()
    
    # Parse configuration
    model_type = parse_am_metadata(args.model_config, 'ModelType') or '2d_schwarzschild'
    mass_param = float(parse_am_metadata(args.model_config, 'MassParameter') or '1.0')
    planck_length = float(parse_am_metadata(args.model_config, 'PlanckLength') or '1e-35')
    bubble_thickness = float(parse_am_metadata(args.model_config, 'BubbleThickness') or '1.0')
    
    print(f"Generating {model_type} model with M={mass_param}, l_p={planck_length}")
    
    results = []
    
    if model_type == '2d_schwarzschild':
        curvature_data = compute_2d_schwarzschild_curvature(mass_param, planck_length)
        geodesic_data = analyze_geodesics(curvature_data, {})
        
        result = {
            'model_type': model_type,
            'curvature_analysis': curvature_data,
            'geodesic_analysis': geodesic_data,
            'planck_scale_analysis': {
                'quantum_dominated': mass_param / planck_length < 1,
                'classical_regime_size': 2 * mass_param,  # Schwarzschild radius
                'quantum_regime_size': planck_length
            }
        }
        results.append(result)
        
    elif model_type == 'warp_bubble':
        curvature_data = compute_warp_bubble_curvature(0.5, bubble_thickness, planck_length)
        geodesic_data = analyze_geodesics(curvature_data, {})
        
        result = {
            'model_type': model_type,
            'curvature_analysis': curvature_data,
            'geodesic_analysis': geodesic_data,
            'planck_scale_analysis': {
                'quantum_dominated': bubble_thickness < planck_length,
                'classical_regime_size': bubble_thickness,
                'quantum_regime_size': planck_length
            }
        }
        results.append(result)
    
    # Write results
    with open(args.out, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    # Write metadata
    summary = {
        'model_generation': 'strong_curvature_toy_models',
        'model_type': model_type,
        'mass_parameter': mass_param,
        'planck_length': planck_length,
        'n_models': len(results),
        'quantum_dominated': results[0]['planck_scale_analysis']['quantum_dominated'] if results else False
    }
    
    with open(args.oam, 'w') as f:
        entries = ', '.join(f'{k} = {v!r}' for k, v in summary.items())
        f.write(f'[ {entries} ]\n')
    
    print(f"Wrote {args.out} ({len(results)} models) and {args.oam}")

if __name__ == '__main__':
    main()
