#!/usr/bin/env python3
"""
Generate minisuperspace cosmological models for strong curvature analysis.
Focus on FRW metrics with Planck-scale curvature.
"""

import argparse
import json
import numpy as np
import sympy as sp
from sympy import symbols, exp, log, sqrt, diff, integrate
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze_sensitivity import parse_am_metadata

def compute_frw_curvature(hubble_param, planck_length, w_eos):
    """
    Compute curvature invariants for FRW cosmology.
    
    Metric: ds² = -dt² + a(t)² [dr² + r²dΩ²]
    Scale factor: a(t) = a₀ (t/t₀)^(2/(3(1+w)))
    """
    t, H0, l_p, w, a0 = symbols('t H0 l_p w a0', real=True, positive=True)
    
    # Scale factor evolution
    if abs(w + 1) > 1e-10:  # Avoid w = -1 (cosmological constant)
        alpha = 2/(3*(1 + w))
        a_t = a0 * (t * H0)**alpha
    else:
        # Exponential expansion for w = -1
        a_t = a0 * exp(H0 * t)
    
    # Hubble parameter H(t) = ȧ/a
    a_dot = diff(a_t, t)
    H_t = a_dot / a_t
    
    # Ricci scalar R = 6(ä/a + (ȧ/a)²)
    a_ddot = diff(a_dot, t)
    ricci_scalar = 6 * (a_ddot/a_t + (a_dot/a_t)**2)
    
    # Weyl tensor vanishes for FRW (conformally flat)
    weyl_scalar = 0
    
    # Kretschmann scalar for FRW
    kretschmann = 12 * H_t**4  # Simplified expression
    
    # Planck time and curvature
    planck_time = l_p  # In natural units where c = 1
    planck_curvature = 1/l_p**2
    
    # Quantum parameter
    quantum_param = kretschmann * l_p**4
    
    return {
        'cosmology_type': 'frw_minisuperspace',
        'scale_factor': str(a_t),
        'hubble_parameter': str(H_t),
        'ricci_scalar': str(ricci_scalar),
        'kretschmann_scalar': str(kretschmann),
        'weyl_scalar': weyl_scalar,
        'quantum_parameter': str(quantum_param),
        'equation_of_state': float(w_eos),
        'planck_time': float(planck_length),
        'initial_hubble': float(hubble_param)
    }

def analyze_horizon_dynamics(cosmology_data):
    """
    Analyze horizon scales and causal structure.
    """
    w = cosmology_data['equation_of_state']
    H0 = cosmology_data['initial_hubble']
    l_p = cosmology_data['planck_time']
    
    # Time evolution
    t_values = np.logspace(-10, 2, 100) * l_p  # From 10⁻¹⁰ to 100 Planck times
    
    # Scale factor evolution
    if abs(w + 1) > 1e-10:
        alpha = 2/(3*(1 + w))
        a_values = (t_values / l_p)**alpha
    else:
        a_values = np.exp(H0 * t_values)
    
    # Hubble parameter
    if abs(w + 1) > 1e-10:
        H_values = alpha / t_values
    else:
        H_values = np.full_like(t_values, H0)
    
    # Horizon scales
    particle_horizon = t_values  # Comoving particle horizon ~ t
    hubble_radius = 1 / H_values  # Hubble radius ~ 1/H
    
    # Quantum regime identification
    quantum_regime = H_values > 1/l_p  # When H > M_Planck
    
    return {
        'time_evolution': t_values.tolist(),
        'scale_factor': a_values.tolist(),
        'hubble_parameter': H_values.tolist(),
        'particle_horizon': particle_horizon.tolist(),
        'hubble_radius': hubble_radius.tolist(),
        'quantum_regime': quantum_regime.tolist(),
        'planck_epoch_duration': float(l_p),
        'classical_epoch_start': float(l_p * 10)  # Rough estimate
    }

def compute_energy_density(cosmology_data, horizon_data):
    """
    Compute energy density evolution and Planck-scale physics.
    """
    w = cosmology_data['equation_of_state']
    H0 = cosmology_data['initial_hubble']
    
    # Energy density from Friedmann equation: ρ = 3H²/(8πG)
    # In Planck units: ρ = 3H²/8π
    H_values = np.array(horizon_data['hubble_parameter'])
    rho_values = 3 * H_values**2 / (8 * np.pi)
    
    # Planck density
    rho_planck = 1.0  # In Planck units
    
    # Pressure from equation of state
    p_values = w * rho_values
    
    return {
        'energy_density': rho_values.tolist(),
        'pressure': p_values.tolist(),
        'planck_density': rho_planck,
        'super_planckian_regime': (rho_values > rho_planck).tolist(),
        'equation_of_state': w,
        'energy_conditions': {
            'null_energy': (rho_values + p_values >= 0).all(),
            'weak_energy': (rho_values >= 0).all() and (rho_values + p_values >= 0).all(),
            'strong_energy': (rho_values + 3*p_values >= 0).all()
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Generate minisuperspace cosmological models")
    parser.add_argument('--cosmo-config', required=True, help="Cosmology configuration (.am)")
    parser.add_argument('--out', required=True, help="Cosmology results (.ndjson)")
    parser.add_argument('--oam', required=True, help="Cosmology metadata (.am)")
    args = parser.parse_args()
    
    # Parse configuration
    hubble_param = float(parse_am_metadata(args.cosmo_config, 'HubbleParameter') or '1.0')
    planck_length = float(parse_am_metadata(args.cosmo_config, 'PlanckLength') or '1e-35')
    w_eos = float(parse_am_metadata(args.cosmo_config, 'EquationOfState') or '0.0')
    
    print(f"Generating FRW cosmology with H₀={hubble_param}, w={w_eos}, l_p={planck_length}")
    
    # Compute cosmological model
    cosmology = compute_frw_curvature(hubble_param, planck_length, w_eos)
    horizon_dynamics = analyze_horizon_dynamics(cosmology)
    energy_evolution = compute_energy_density(cosmology, horizon_dynamics)
    
    result = {
        'model_type': 'frw_minisuperspace',
        'cosmological_parameters': cosmology,
        'horizon_dynamics': horizon_dynamics,
        'energy_evolution': energy_evolution,
        'planck_scale_analysis': {
            'quantum_gravity_epoch': any(horizon_dynamics['quantum_regime']),
            'super_planckian_density': any(energy_evolution['super_planckian_regime']),
            'classical_transition_time': horizon_dynamics['classical_epoch_start'],
            'energy_conditions_satisfied': energy_evolution['energy_conditions']
        }
    }
    
    # Write results
    with open(args.out, 'w') as f:
        f.write(json.dumps(result) + '\n')
    
    # Write metadata
    summary = {
        'model_generation': 'minisuperspace_cosmology',
        'cosmology_type': 'frw',
        'hubble_parameter': hubble_param,
        'equation_of_state': w_eos,
        'planck_length': planck_length,
        'quantum_epoch': result['planck_scale_analysis']['quantum_gravity_epoch'],
        'super_planckian': result['planck_scale_analysis']['super_planckian_density']
    }
    
    with open(args.oam, 'w') as f:
        entries = ', '.join(f'{k} = {v!r}' for k, v in summary.items())
        f.write(f'[ {entries} ]\n')
    
    print(f"Wrote {args.out} (1 cosmological model) and {args.oam}")

if __name__ == '__main__':
    main()
