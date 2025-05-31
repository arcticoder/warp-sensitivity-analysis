# Consistency Check Framework for Warp Sensitivity Analysis

This framework provides automated verification of theoretical consistency requirements for quantum gravity corrections in the warp sensitivity analysis project.

## Purpose

The consistency checks validate three fundamental theoretical requirements:

1. **Unitarity** - Quantum probability conservation and positivity
2. **Gauge Invariance** - Independence from coordinate/gauge choices
3. **Classical Limit** - Recovery of General Relativity as ℏ→0

## Directory Structure

```
consistency_checks/
├── consistency_config.am       # Configuration for tests and tolerances
├── verify_unitarity.py         # Unitarity validation
├── verify_gauge_invariance.py  # Gauge independence verification
├── recover_classical_limit.py  # ℏ→0 limit tests
├── run_consistency_checks.py   # Comprehensive test runner
├── reference_solutions/        # Reference data for validation
│   ├── schwarzschild_limit.ndjson  # Classical GR solutions
│   └── gauge_test_cases.am        # Gauge transformation definitions
└── outputs/                    # Test results and reports
    ├── consistency_report.ndjson   # Detailed JSON test results
    ├── consistency_report.am       # Summary in AsciiMath format
    ├── gauge_test.am              # Gauge invariance summary
    ├── gauge_test.ndjson          # Detailed gauge test results
    ├── unitarity_test.am          # Unitarity test summary
    ├── unitarity_test.ndjson      # Detailed unitarity results
    ├── classical_limit_test.am    # Classical limit summary
    └── classical_limit_test.ndjson # Classical limit test details
```

## Key Components

### 1. Unitarity Verification

The `verify_unitarity.py` script checks that quantum gravity corrections preserve:
- Probability conservation for quantum states
- Positivity of two-point correlation functions
- S-matrix unitarity for scattering processes
- Trace preservation for density matrices

### 2. Gauge Invariance Testing

The `verify_gauge_invariance.py` script applies gauge transformations to verify:
- Diffeomorphism invariance (coordinate independence)
- Gauge parameter independence
- BRST symmetry preservation
- Background independence

### 3. Classical Limit Recovery

The `recover_classical_limit.py` script validates that:
- Setting ℏ→0 recovers classical General Relativity
- Quantum corrections scale appropriately with powers of ℏ
- The Schwarzschild metric and other classical solutions are recovered
- Linearized waveforms match classical expectations

## Configuration

The `consistency_config.am` file configures:
- Numerical tolerances for each test type
- Which tests to enable/disable
- Input data sources from semi-classical and strong-curvature analyses
- Reference solution paths
- Output file locations

## Running Tests

The comprehensive test suite can be executed with:

```
python run_consistency_checks.py --config consistency_config.am
```

Individual tests can also be run separately:

```
python verify_unitarity.py --config consistency_config.am --out outputs/unitarity_test.ndjson --oam outputs/unitarity_test.am
python verify_gauge_invariance.py --config consistency_config.am --out outputs/gauge_test.ndjson --oam outputs/gauge_test.am
python recover_classical_limit.py --config consistency_config.am --out outputs/classical_limit_test.ndjson --oam outputs/classical_limit_test.am
```

## Results

Test results are output in two formats:
1. **NDJSON files** - Detailed test results for programmatic analysis
2. **AsciiMath (.am) files** - Human-readable summaries following project conventions

The overall consistency status is provided in `consistency_report.am`, with a 
detailed breakdown of all tests in `consistency_report.ndjson`.
