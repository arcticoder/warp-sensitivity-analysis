#!/usr/bin/env python3
"""
Compare and analyze strong curvature regime models.
Unifies outputs from different toy models and identifies quantum-dominated regimes.
"""

import argparse
import json
import numpy as np
import ndjson
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze_sensitivity import parse_am_metadata

def load_model_data(ndjson_path):
    """Load model data from JSON-lines file."""
    with open(ndjson_path, 'r') as f:
        return ndjson.load(f)

def classify_regime(model_data):
    """
    Classify whether model is in classical, semi-classical, or quantum regime.
    """
    if model_data['model_type'] == '2d_schwarzschild':
        quantum_dominated = model_data['planck_scale_analysis']['quantum_dominated']
        classical_size = model_data['planck_scale_analysis']['classical_regime_size']
        quantum_size = model_data['planck_scale_analysis']['quantum_regime_size']
        
        if quantum_dominated:
            return 'quantum_gravity'
        elif classical_size < 10 * quantum_size:
            return 'semi_classical'
        else:
            return 'classical'
            
    elif model_data['model_type'] == 'warp_bubble':
        quantum_dominated = model_data['planck_scale_analysis']['quantum_dominated']
        thickness = model_data['planck_scale_analysis']['classical_regime_size']
        planck_scale = model_data['planck_scale_analysis']['quantum_regime_size']
        
        if quantum_dominated:
            return 'quantum_gravity'
        elif thickness < 100 * planck_scale:
            return 'semi_classical'
        else:
            return 'classical'
            
    elif model_data['model_type'] == 'frw_minisuperspace':
        quantum_epoch = model_data['planck_scale_analysis']['quantum_gravity_epoch']
        super_planckian = model_data['planck_scale_analysis']['super_planckian_density']
        
        if quantum_epoch and super_planckian:
            return 'quantum_gravity'
        elif quantum_epoch or super_planckian:
            return 'semi_classical'
        else:
            return 'classical'
    
    return 'unknown'

def extract_curvature_scales(model_data):
    """Extract characteristic curvature scales from model."""
    if 'curvature_analysis' in model_data:
        curvature_data = model_data['curvature_analysis']
        
        # Try to extract numerical curvature scale
        if 'quantum_parameter' in curvature_data:
            try:
                # This is symbolic, but we can estimate
                if model_data['model_type'] == '2d_schwarzschild':
                    M = curvature_data['mass_parameter']
                    l_p = curvature_data['planck_length']
                    # Kretschmann ~ M²/r⁶, at horizon r ~ M
                    curvature_scale = M**2 / M**6  # Simplified
                    quantum_param = curvature_scale * l_p**4
                    return {
                        'curvature_scale': float(1/M**4),
                        'quantum_parameter': float(quantum_param),
                        'planck_curvature': float(1/l_p**2)
                    }
                elif model_data['model_type'] == 'warp_bubble':
                    thickness = curvature_data['bubble_thickness']
                    l_p = curvature_data['planck_length']
                    curvature_scale = 1/thickness**2
                    quantum_param = (thickness/l_p)**(-2)
                    return {
                        'curvature_scale': float(curvature_scale),
                        'quantum_parameter': float(quantum_param),
                        'planck_curvature': float(1/l_p**2)
                    }
            except:
                pass
    
    # Fallback for cosmological models
    if model_data['model_type'] == 'frw_minisuperspace':
        cosmo_params = model_data['cosmological_parameters']
        H0 = cosmo_params['initial_hubble']
        l_p = cosmo_params['planck_time']
        
        return {
            'curvature_scale': float(H0**2),
            'quantum_parameter': float((H0 * l_p)**2),
            'planck_curvature': float(1/l_p**2)
        }
    
    return {'curvature_scale': 0, 'quantum_parameter': 0, 'planck_curvature': 1}

def compare_models(model_list):
    """Compare multiple strong curvature models."""
    comparison_results = []
    
    for i, model in enumerate(model_list):
        regime = classify_regime(model)
        curvature_info = extract_curvature_scales(model)
        
        # Assess validity of classical GR
        quantum_param = curvature_info['quantum_parameter']
        classical_valid = quantum_param < 0.1  # Classical when quantum corrections < 10%
        
        # Estimate where quantum corrections become important
        if quantum_param > 0:
            quantum_correction_estimate = min(quantum_param, 1.0)  # Cap at 100%
        else:
            quantum_correction_estimate = 0
        
        result = {
            'model_id': i,
            'model_type': model['model_type'],
            'regime_classification': regime,
            'curvature_analysis': curvature_info,
            'classical_gr_valid': classical_valid,
            'quantum_correction_strength': quantum_correction_estimate,
            'requires_quantum_gravity': regime == 'quantum_gravity',
            'parameter_ranges': extract_parameter_ranges(model)
        }
        
        comparison_results.append(result)
    
    return comparison_results

def extract_parameter_ranges(model_data):
    """Extract valid parameter ranges for each model."""
    if model_data['model_type'] == '2d_schwarzschild':
        M = model_data.get('curvature_analysis', {}).get('mass_parameter', 1.0)
        l_p = model_data.get('curvature_analysis', {}).get('planck_length', 1e-35)
        
        return {
            'mass_parameter': [l_p, 100*l_p],  # Range where model is meaningful
            'classical_valid_range': [10*l_p, float('inf')],
            'quantum_regime_range': [0, l_p]
        }
        
    elif model_data['model_type'] == 'warp_bubble':
        thickness = model_data.get('curvature_analysis', {}).get('bubble_thickness', 1.0)
        l_p = model_data.get('curvature_analysis', {}).get('planck_length', 1e-35)
        
        return {
            'bubble_thickness': [l_p, 1000*l_p],
            'classical_valid_range': [100*l_p, float('inf')],
            'quantum_regime_range': [0, l_p]
        }
        
    elif model_data['model_type'] == 'frw_minisuperspace':
        H0 = model_data.get('cosmological_parameters', {}).get('initial_hubble', 1.0)
        l_p = model_data.get('cosmological_parameters', {}).get('planck_time', 1e-35)
        
        return {
            'hubble_parameter': [1/l_p/1000, 1/l_p],  # From sub-Planckian to Planckian
            'classical_valid_range': [0, 1/l_p/10],
            'quantum_regime_range': [1/l_p/10, 1/l_p]
        }
    
    return {}

def main():
    parser = argparse.ArgumentParser(description="Compare strong curvature models")
    parser.add_argument('--models', nargs='+', required=True, 
                       help="Model data files (.ndjson)")
    parser.add_argument('--meta', nargs='+', required=True,
                       help="Model metadata files (.am)")
    parser.add_argument('--out', required=True, help="Comparison results (.ndjson)")
    parser.add_argument('--oam', required=True, help="Comparison summary (.am)")
    args = parser.parse_args()
    
    print(f"Comparing {len(args.models)} strong curvature models...")
    
    # Load all models
    all_models = []
    model_types = []
    
    for model_file, meta_file in zip(args.models, args.meta):
        models = load_model_data(model_file)
        model_type = parse_am_metadata(meta_file, 'model_type') or 'unknown'
        
        all_models.extend(models)
        model_types.extend([model_type] * len(models))
    
    # Compare models
    comparison_results = compare_models(all_models)
    
    # Write detailed results
    with open(args.out, 'w') as f:
        for result in comparison_results:
            f.write(json.dumps(result) + '\n')
    
    # Generate summary statistics
    regime_counts = {}
    for result in comparison_results:
        regime = result['regime_classification']
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
    
    quantum_gravity_models = [r for r in comparison_results if r['requires_quantum_gravity']]
    classical_models = [r for r in comparison_results if r['classical_gr_valid']]
    
    summary = {
        'comparison_type': 'strong_curvature_models',
        'n_models_total': len(comparison_results),
        'model_types': list(set(model_types)),
        'regime_distribution': regime_counts,
        'n_quantum_gravity': len(quantum_gravity_models),
        'n_classical_valid': len(classical_models),
        'average_quantum_correction': float(np.mean([r['quantum_correction_strength'] 
                                                   for r in comparison_results])),
        'planck_scale_physics_important': len(quantum_gravity_models) > 0
    }
    
    with open(args.oam, 'w') as f:
        entries = ', '.join(f'{k} = {v!r}' for k, v in summary.items())
        f.write(f'[ {entries} ]\n')
    
    print(f"Wrote {args.out} ({len(comparison_results)} comparisons) and {args.oam}")
    print(f"Regime distribution: {regime_counts}")
    print(f"Models requiring quantum gravity: {len(quantum_gravity_models)}")

if __name__ == '__main__':
    main()
