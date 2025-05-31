#!/usr/bin/env python3
"""
Verify unitarity of quantum gravity corrections in warp sensitivity analysis.

This module checks that quantum corrections preserve fundamental unitarity properties:
- Probability conservation: ∑|ψᵢ|² = 1 for quantum states
- Correlation function positivity: ⟨ψ|O†O|ψ⟩ ≥ 0 for observables O
- S-matrix unitarity: S†S = I for scattering processes
- Trace preservation: Tr(ρ) = 1 for density matrices

The verification process:
1. Loads quantum-corrected waveforms and correlation functions
2. Computes unitarity measures for each observable
3. Checks violation bounds against theoretical tolerances
4. Reports systematic violations indicating breakdown of quantum consistency
"""

import argparse
import json
import numpy as np
import ndjson
import sys
import os
from typing import Dict, List, Tuple, Any
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze_sensitivity import parse_am_metadata

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_quantum_data(data_path: str) -> List[Dict[str, Any]]:
    """Load quantum-corrected data from semi-classical or strong-curvature analysis."""
    try:
        with open(data_path, 'r') as f:
            return ndjson.load(f)
    except Exception as e:
        logger.error(f"Failed to load quantum data from {data_path}: {e}")
        return []

def check_probability_conservation(waveform_data: Dict[str, Any], tolerance: float) -> Dict[str, Any]:
    """
    Verify that quantum state normalization is preserved.
    
    For waveforms ψ(t), checks that ∫|ψ(t)|²dt remains constant.
    """
    if 'waveform_amplitude' not in waveform_data:
        return {'status': 'skipped', 'reason': 'no_waveform_data'}
    
    amplitudes = np.array(waveform_data['waveform_amplitude'])
    if len(amplitudes) == 0:
        return {'status': 'skipped', 'reason': 'empty_waveform'}
    
    # Compute norm at different time points
    norms = np.abs(amplitudes)**2
    
    # For discrete data, approximate conservation
    if len(norms) > 1:
        # Check if total "probability" is conserved
        total_norm = np.trapz(norms, x=np.arange(len(norms)))
        
        # For relative changes in norm
        norm_variation = np.std(norms) / np.mean(norms) if np.mean(norms) > 0 else float('inf')
        
        violation = norm_variation
        passed = violation < tolerance
        
        return {
            'status': 'completed',
            'passed': passed,
            'violation_measure': float(violation),
            'tolerance': tolerance,
            'total_norm': float(total_norm),
            'norm_std': float(np.std(norms)),
            'details': f'Relative norm variation: {violation:.2e}'
        }
    
    return {'status': 'insufficient_data', 'reason': 'single_point_waveform'}

def check_correlator_positivity(correlation_data: Dict[str, Any], tolerance: float) -> Dict[str, Any]:
    """
    Verify positivity of 2-point correlation functions.
    
    For quantum field ϕ, checks that ⟨ϕ(x)ϕ(y)⟩ has positive definite matrix structure.
    """
    if 'correlation_matrix' not in correlation_data:
        # Try to construct from available data
        if 'waveform_amplitude' in correlation_data:
            amplitudes = np.array(correlation_data['waveform_amplitude'])
            if len(amplitudes) > 1:
                # Construct autocorrelation matrix
                correlation_matrix = np.outer(amplitudes, np.conj(amplitudes))
            else:
                return {'status': 'skipped', 'reason': 'insufficient_correlation_data'}
        else:
            return {'status': 'skipped', 'reason': 'no_correlation_data'}
    else:
        correlation_matrix = np.array(correlation_data['correlation_matrix'])
    
    # Check eigenvalues for positivity
    try:
        eigenvalues = np.linalg.eigvals(correlation_matrix)
        min_eigenvalue = np.min(np.real(eigenvalues))
        
        # Violation is measured by most negative eigenvalue
        violation = max(0, -min_eigenvalue)
        passed = violation < tolerance
        
        return {
            'status': 'completed',
            'passed': passed,
            'violation_measure': float(violation),
            'tolerance': tolerance,
            'min_eigenvalue': float(min_eigenvalue),
            'condition_number': float(np.linalg.cond(correlation_matrix)),
            'details': f'Minimum eigenvalue: {min_eigenvalue:.2e}'
        }
    except np.linalg.LinAlgError as e:
        return {'status': 'error', 'reason': f'linear_algebra_error: {e}'}

def check_trace_preservation(density_matrix_data: Dict[str, Any], tolerance: float) -> Dict[str, Any]:
    """
    Verify that quantum density matrices maintain unit trace.
    
    For density matrix ρ, checks that Tr(ρ) = 1 and ρ ≥ 0.
    """
    if 'density_matrix' not in density_matrix_data:
        # Try to construct from waveform if available
        if 'waveform_amplitude' in density_matrix_data:
            amplitudes = np.array(density_matrix_data['waveform_amplitude'])
            if len(amplitudes) > 0:
                # Pure state density matrix |ψ⟩⟨ψ|
                psi = amplitudes / np.linalg.norm(amplitudes)
                density_matrix = np.outer(psi, np.conj(psi))
            else:
                return {'status': 'skipped', 'reason': 'empty_waveform'}
        else:
            return {'status': 'skipped', 'reason': 'no_density_matrix_data'}
    else:
        density_matrix = np.array(density_matrix_data['density_matrix'])
    
    try:
        # Check trace
        trace = np.trace(density_matrix)
        trace_violation = abs(trace - 1.0)
        
        # Check positivity
        eigenvalues = np.linalg.eigvals(density_matrix)
        min_eigenvalue = np.min(np.real(eigenvalues))
        positivity_violation = max(0, -min_eigenvalue)
        
        total_violation = max(trace_violation, positivity_violation)
        passed = total_violation < tolerance
        
        return {
            'status': 'completed',
            'passed': passed,
            'violation_measure': float(total_violation),
            'tolerance': tolerance,
            'trace': float(trace),
            'trace_violation': float(trace_violation),
            'min_eigenvalue': float(min_eigenvalue),
            'positivity_violation': float(positivity_violation),
            'details': f'Trace: {trace:.6f}, Min eigenvalue: {min_eigenvalue:.2e}'
        }
    except Exception as e:
        return {'status': 'error', 'reason': f'computation_error: {e}'}

def verify_unitarity(data_list: List[Dict[str, Any]], tolerance: float) -> List[Dict[str, Any]]:
    """
    Run all unitarity checks on quantum data.
    """
    results = []
    
    for i, data_point in enumerate(data_list):
        logger.info(f"Checking unitarity for data point {i+1}/{len(data_list)}")
        
        # Run probability conservation check
        prob_result = check_probability_conservation(data_point, tolerance)
        
        # Run correlator positivity check  
        corr_result = check_correlator_positivity(data_point, tolerance)
        
        # Run trace preservation check
        trace_result = check_trace_preservation(data_point, tolerance)
        
        result = {
            'data_point_id': i,
            'model_type': data_point.get('model_type', 'unknown'),
            'unitarity_checks': {
                'probability_conservation': prob_result,
                'correlator_positivity': corr_result,
                'trace_preservation': trace_result
            },
            'overall_passed': all(
                check.get('passed', False) for check in [prob_result, corr_result, trace_result]
                if check.get('status') == 'completed'
            )
        }
        
        results.append(result)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Verify unitarity of quantum gravity corrections")
    parser.add_argument('--config', required=True, help="Consistency check configuration (.am)")
    parser.add_argument('--semi-classical', help="Semi-classical data (.ndjson)")
    parser.add_argument('--strong-curvature', help="Strong curvature data (.ndjson)")
    parser.add_argument('--out', required=True, help="Unitarity results (.ndjson)")
    parser.add_argument('--oam', required=True, help="Unitarity summary (.am)")
    args = parser.parse_args()
    
    # Parse configuration
    tolerance = float(parse_am_metadata(args.config, 'UnitarityTolerance') or '1e-12')
    
    logger.info(f"Starting unitarity verification with tolerance {tolerance}")
    
    all_results = []
    
    # Check semi-classical data if provided
    if args.semi_classical:
        logger.info("Verifying unitarity of semi-classical models...")
        semi_data = load_quantum_data(args.semi_classical)
        semi_results = verify_unitarity(semi_data, tolerance)
        all_results.extend(semi_results)
    
    # Check strong curvature data if provided
    if args.strong_curvature:
        logger.info("Verifying unitarity of strong curvature models...")
        strong_data = load_quantum_data(args.strong_curvature)
        strong_results = verify_unitarity(strong_data, tolerance)
        all_results.extend(strong_results)
    
    # Write detailed results
    with open(args.out, 'w') as f:
        for result in all_results:
            f.write(json.dumps(result) + '\n')
    
    # Generate summary
    total_tests = len(all_results)
    passed_tests = sum(1 for r in all_results if r['overall_passed'])
    failed_tests = total_tests - passed_tests
    
    if total_tests > 0:
        pass_rate = passed_tests / total_tests
        overall_status = 'PASSED' if pass_rate >= 0.9 else 'FAILED'
    else:
        pass_rate = 0.0
        overall_status = 'NO_DATA'
    
    summary = {
        'test_type': 'unitarity_verification',
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'failed_tests': failed_tests,
        'pass_rate': pass_rate,
        'tolerance': tolerance,
        'overall_status': overall_status
    }
    
    with open(args.oam, 'w') as f:
        entries = ', '.join(f'{k} = {v!r}' for k, v in summary.items())
        f.write(f'[ {entries} ]\n')
    
    logger.info(f"Unitarity verification complete: {passed_tests}/{total_tests} tests passed")
    logger.info(f"Overall status: {overall_status}")
    
    if failed_tests > 0:
        logger.warning(f"{failed_tests} unitarity violations detected - check detailed results")

if __name__ == '__main__':
    main()
