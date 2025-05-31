#!/usr/bin/env python3
"""
Comprehensive consistency check runner for warp sensitivity analysis.

This script orchestrates all consistency verifications:
- Unitarity preservation in quantum corrections
- Gauge invariance of physical observables  
- Classical limit recovery as ℏ→0
- Overall theoretical consistency assessment

Usage:
    python run_consistency_checks.py --config consistency_config.am
"""

import argparse
import json
import subprocess
import sys
import os
from pathlib import Path
import logging
from typing import Dict, List, Any
import ndjson

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze_sensitivity import parse_am_metadata

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConsistencyCheckRunner:
    """Orchestrates all consistency verification tests."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.results = {}
        
    def _load_config(self) -> Dict[str, Any]:
        """Load consistency check configuration."""
        config = {}
        
        def clean_parse(key: str, default: str) -> str:
            """Parse and clean AsciiMath metadata value."""
            value = parse_am_metadata(self.config_path, key) or default
            # Remove comments and whitespace
            if '#' in value:
                value = value.split('#')[0]
            return value.strip()
        
        # Parse AsciiMath configuration
        config['unitarity_tolerance'] = float(clean_parse('UnitarityTolerance', '1e-12'))
        config['gauge_tolerance'] = float(clean_parse('GaugeInvarianceTolerance', '1e-10'))
        config['classical_tolerance'] = float(clean_parse('ClassicalLimitTolerance', '1e-8'))
        
        config['enable_unitarity'] = clean_parse('EnableUnitarityCheck', 'true') == 'true'
        config['enable_gauge'] = clean_parse('EnableGaugeInvarianceCheck', 'true') == 'true'
        config['enable_classical'] = clean_parse('EnableClassicalLimitCheck', 'true') == 'true'
        
        config['semi_classical_data'] = clean_parse('SemiClassicalDataPath', '')
        config['strong_curvature_data'] = clean_parse('StrongCurvatureDataPath', '')
        config['schwarzschild_reference'] = clean_parse('SchwarzschildReferencePath', '')
        config['gauge_test_cases'] = clean_parse('GaugeTestCasesPath', '')
        
        config['detailed_report'] = clean_parse('DetailedReportPath', '')
        config['summary_report'] = clean_parse('SummaryReportPath', '')
        config['failure_threshold'] = float(clean_parse('FailureThreshold', '0.1'))
        
        return config
    
    def _resolve_path(self, relative_path: str) -> str:
        """Resolve path relative to config directory."""
        if not relative_path:
            return ""
        
        config_dir = os.path.dirname(self.config_path)
        return os.path.join(config_dir, relative_path)
    
    def run_unitarity_check(self) -> Dict[str, Any]:
        """Run unitarity verification tests."""
        if not self.config['enable_unitarity']:
            return {'status': 'skipped', 'reason': 'disabled_in_config'}
        
        logger.info("Running unitarity verification...")
        
        script_path = os.path.join(os.path.dirname(__file__), 'verify_unitarity.py')
        output_path = self._resolve_path('outputs/unitarity_test.ndjson')
        summary_path = self._resolve_path('outputs/unitarity_test.am')
        
        cmd = [
            sys.executable, script_path,
            '--config', self.config_path,
            '--out', output_path,
            '--oam', summary_path
        ]
        
        # Add data sources if available
        if self.config['semi_classical_data']:
            semi_path = self._resolve_path(self.config['semi_classical_data'])
            if os.path.exists(semi_path):
                cmd.extend(['--semi-classical', semi_path])
        
        if self.config['strong_curvature_data']:
            strong_path = self._resolve_path(self.config['strong_curvature_data'])
            if os.path.exists(strong_path):
                cmd.extend(['--strong-curvature', strong_path])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                # Parse summary results
                summary_data = parse_am_metadata(summary_path, '')
                return {
                    'status': 'completed',
                    'overall_status': summary_data.get('overall_status', 'UNKNOWN'),
                    'pass_rate': float(summary_data.get('pass_rate', 0.0)),
                    'total_tests': int(summary_data.get('total_tests', 0)),
                    'tolerance': self.config['unitarity_tolerance']
                }
            else:
                logger.error(f"Unitarity check failed: {result.stderr}")
                return {'status': 'failed', 'error': result.stderr}
                
        except Exception as e:
            logger.error(f"Error running unitarity check: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def run_gauge_invariance_check(self) -> Dict[str, Any]:
        """Run gauge invariance verification tests."""
        if not self.config['enable_gauge']:
            return {'status': 'skipped', 'reason': 'disabled_in_config'}
        
        logger.info("Running gauge invariance verification...")
        
        script_path = os.path.join(os.path.dirname(__file__), 'verify_gauge_invariance.py')
        output_path = self._resolve_path('outputs/gauge_test.ndjson')
        summary_path = self._resolve_path('outputs/gauge_test.am')
        
        cmd = [
            sys.executable, script_path,
            '--config', self.config_path,
            '--out', output_path,
            '--oam', summary_path
        ]
        
        # Add test cases if available
        if self.config['gauge_test_cases']:
            cases_path = self._resolve_path(self.config['gauge_test_cases'])
            if os.path.exists(cases_path):
                cmd.extend(['--test-cases', cases_path])
        
        # Add data sources
        if self.config['semi_classical_data']:
            semi_path = self._resolve_path(self.config['semi_classical_data'])
            if os.path.exists(semi_path):
                cmd.extend(['--semi-classical', semi_path])
        
        if self.config['strong_curvature_data']:
            strong_path = self._resolve_path(self.config['strong_curvature_data'])
            if os.path.exists(strong_path):
                cmd.extend(['--strong-curvature', strong_path])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                summary_data = parse_am_metadata(summary_path, '')
                return {
                    'status': 'completed',
                    'overall_status': summary_data.get('overall_status', 'UNKNOWN'),
                    'pass_rate': float(summary_data.get('pass_rate', 0.0)),
                    'total_transformations': int(summary_data.get('total_transformations', 0)),
                    'tolerance': self.config['gauge_tolerance']
                }
            else:
                logger.error(f"Gauge invariance check failed: {result.stderr}")
                return {'status': 'failed', 'error': result.stderr}
                
        except Exception as e:
            logger.error(f"Error running gauge invariance check: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def run_classical_limit_check(self) -> Dict[str, Any]:
        """Run classical limit recovery verification."""
        if not self.config['enable_classical']:
            return {'status': 'skipped', 'reason': 'disabled_in_config'}
        
        logger.info("Running classical limit verification...")
        
        script_path = os.path.join(os.path.dirname(__file__), 'recover_classical_limit.py')
        output_path = self._resolve_path('outputs/classical_limit_test.ndjson')
        summary_path = self._resolve_path('outputs/classical_limit_test.am')
        
        cmd = [
            sys.executable, script_path,
            '--config', self.config_path,
            '--out', output_path,
            '--oam', summary_path
        ]
        
        # Add reference solutions if available
        if self.config['schwarzschild_reference']:
            ref_path = self._resolve_path(self.config['schwarzschild_reference'])
            if os.path.exists(ref_path):
                cmd.extend(['--reference', ref_path])
        
        # Add data sources
        if self.config['semi_classical_data']:
            semi_path = self._resolve_path(self.config['semi_classical_data'])
            if os.path.exists(semi_path):
                cmd.extend(['--semi-classical', semi_path])
        
        if self.config['strong_curvature_data']:
            strong_path = self._resolve_path(self.config['strong_curvature_data'])
            if os.path.exists(strong_path):
                cmd.extend(['--strong-curvature', strong_path])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            
            if result.returncode == 0:
                summary_data = parse_am_metadata(summary_path, '')
                return {
                    'status': 'completed',
                    'overall_status': summary_data.get('overall_status', 'UNKNOWN'),
                    'pass_rate': float(summary_data.get('pass_rate', 0.0)),
                    'total_tests': int(summary_data.get('total_tests', 0)),
                    'tolerance': self.config['classical_tolerance']
                }
            else:
                logger.error(f"Classical limit check failed: {result.stderr}")
                return {'status': 'failed', 'error': result.stderr}
                
        except Exception as e:
            logger.error(f"Error running classical limit check: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all enabled consistency checks."""
        logger.info("Starting comprehensive consistency verification...")
        
        # Run individual checks
        self.results['unitarity'] = self.run_unitarity_check()
        self.results['gauge_invariance'] = self.run_gauge_invariance_check()
        self.results['classical_limit'] = self.run_classical_limit_check()
        
        # Compute overall assessment
        overall_result = self._assess_overall_consistency()
        self.results['overall_assessment'] = overall_result
        
        return self.results
    
    def _assess_overall_consistency(self) -> Dict[str, Any]:
        """Assess overall theoretical consistency."""
        completed_checks = []
        passed_checks = []
        total_tests = 0
        passed_tests = 0
        
        for check_name, result in self.results.items():
            if result.get('status') == 'completed':
                completed_checks.append(check_name)
                
                if result.get('overall_status') == 'PASSED':
                    passed_checks.append(check_name)
                
                total_tests += result.get('total_tests', 0)
                passed_tests += int(result.get('total_tests', 0) * result.get('pass_rate', 0.0))
        
        if len(completed_checks) == 0:
            overall_status = 'NO_DATA'
            overall_pass_rate = 0.0
        else:
            overall_pass_rate = len(passed_checks) / len(completed_checks)
            
            if overall_pass_rate >= (1.0 - self.config['failure_threshold']):
                overall_status = 'PASSED'
            else:
                overall_status = 'FAILED'
        
        return {
            'status': 'completed',
            'overall_status': overall_status,
            'checks_completed': len(completed_checks),
            'checks_passed': len(passed_checks),
            'overall_pass_rate': overall_pass_rate,
            'total_individual_tests': total_tests,
            'passed_individual_tests': passed_tests,
            'failure_threshold': self.config['failure_threshold'],
            'theoretical_consistency': overall_status == 'PASSED'
        }
    
    def generate_reports(self):
        """Generate detailed and summary reports."""
        # Detailed JSON report
        if self.config['detailed_report']:
            detailed_path = self._resolve_path(self.config['detailed_report'])
            try:
                with open(detailed_path, 'w') as f:
                    json.dump(self.results, f, indent=2)
                logger.info(f"Detailed report written to {detailed_path}")
            except Exception as e:
                logger.error(f"Failed to write detailed report: {e}")
        
        # AsciiMath summary report
        if self.config['summary_report']:
            summary_path = self._resolve_path(self.config['summary_report'])
            try:
                self._write_summary_report(summary_path)
                logger.info(f"Summary report written to {summary_path}")
            except Exception as e:
                logger.error(f"Failed to write summary report: {e}")
    
    def _write_summary_report(self, summary_path: str):
        """Write AsciiMath summary report."""
        overall = self.results.get('overall_assessment', {})
        
        summary_data = {
            'test_suite': 'comprehensive_consistency_verification',
            'overall_status': overall.get('overall_status', 'UNKNOWN'),
            'theoretical_consistency': overall.get('theoretical_consistency', False),
            'checks_completed': overall.get('checks_completed', 0),
            'checks_passed': overall.get('checks_passed', 0),
            'overall_pass_rate': overall.get('overall_pass_rate', 0.0),
            'total_individual_tests': overall.get('total_individual_tests', 0),
            'passed_individual_tests': overall.get('passed_individual_tests', 0),
            'failure_threshold': overall.get('failure_threshold', 0.1)
        }
        
        # Add individual check results
        for check_name, result in self.results.items():
            if check_name != 'overall_assessment' and result.get('status') == 'completed':
                summary_data[f'{check_name}_status'] = result.get('overall_status', 'UNKNOWN')
                summary_data[f'{check_name}_pass_rate'] = result.get('pass_rate', 0.0)
        
        with open(summary_path, 'w') as f:
            entries = ', '.join(f'{k} = {v!r}' for k, v in summary_data.items())
            f.write(f'[ {entries} ]\n')


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive consistency checks")
    parser.add_argument('--config', required=True, help="Consistency check configuration (.am)")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Run consistency checks
    runner = ConsistencyCheckRunner(args.config)
    results = runner.run_all_checks()
    
    # Generate reports
    runner.generate_reports()
    
    # Print summary
    overall = results.get('overall_assessment', {})
    status = overall.get('overall_status', 'UNKNOWN')
    
    print(f"\n{'='*60}")
    print(f"CONSISTENCY CHECK RESULTS")
    print(f"{'='*60}")
    print(f"Overall Status: {status}")
    print(f"Theoretical Consistency: {'YES' if overall.get('theoretical_consistency', False) else 'NO'}")
    print(f"Checks Completed: {overall.get('checks_completed', 0)}")
    print(f"Checks Passed: {overall.get('checks_passed', 0)}")
    print(f"Overall Pass Rate: {overall.get('overall_pass_rate', 0.0):.1%}")
    print(f"{'='*60}")
    
    # Individual check summaries
    for check_name, result in results.items():
        if check_name != 'overall_assessment':
            if result.get('status') == 'completed':
                check_status = result.get('overall_status', 'UNKNOWN')
                pass_rate = result.get('pass_rate', 0.0)
                print(f"{check_name.replace('_', ' ').title()}: {check_status} ({pass_rate:.1%})")
            elif result.get('status') == 'skipped':
                print(f"{check_name.replace('_', ' ').title()}: SKIPPED")
            else:
                print(f"{check_name.replace('_', ' ').title()}: ERROR")
    
    print(f"{'='*60}")
    
    # Exit with appropriate code
    if status == 'PASSED':
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
