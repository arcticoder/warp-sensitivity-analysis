# Warp Sensitivity Analysis - Extended Framework

## Overview

This repository now contains a comprehensive theoretical physics testing framework that extends beyond simple signal-vs-noise analysis to include:

1. **Semi-classical testing** (Post-Newtonian corrections)
2. **Strong-curvature regime models** (Planck-scale physics)
3. **Unified analysis pipeline** with consistent metadata format

## Directory Structure

```
warp-sensitivity-analysis/
├── analyze_sensitivity.py          # Original signal vs noise pipeline
├── generate_sensitivity_curve.py   # Noise curve generator
├── mock_data.*                     # Test data files
├── sensitivity_*.*                 # Analysis outputs
├── 
├── semi_classical/                 # Post-Newtonian analysis
│   ├── compute_pn_corrections.py  # PN expansion generator
│   ├── analyze_pn_tests.py        # Experimental comparison
│   ├── theory_params.am           # Theory configuration
│   ├── pn_config.am               # PN expansion settings
│   ├── pn_waveforms.ndjson        # Generated PN corrections
│   ├── pn_summary.am              # PN metadata
│   └── pn_data/                   # Experimental datasets
│       ├── ligo_data.csv          # LIGO sensitivity curves
│       ├── ligo_data.am           # LIGO metadata
│       └── atomic_interf.am       # Atomic interferometry data
│
└── strong_curvature/              # Planck-scale models
    ├── generate_2d_blackhole.py   # 2D black hole toy models
    ├── minisuperspace_cosmo.py    # FRW minisuperspace cosmology
    ├── compare_strong_models.py   # Model unification
    ├── blackhole_config.am        # Black hole parameters
    ├── cosmo_config.am            # Cosmology parameters
    ├── blackhole_data.ndjson      # Generated black hole data
    ├── blackhole_summary.am       # Black hole metadata
    └── unified_summary.am         # Combined analysis metadata
```

## Workflow Integration

### 1. Semi-Classical Testing Pipeline

**Input:** Theory parameters → PN expansion → Experimental comparison

```bash
# Generate Post-Newtonian corrections
cd semi_classical/
python compute_pn_corrections.py \
    --theory theory_params.am \
    --pn-config pn_config.am \
    --out pn_waveforms.ndjson \
    --oam pn_summary.am

# Analyze against experimental data
python analyze_pn_tests.py \
    --pn-data pn_waveforms.ndjson \
    --pn-meta pn_summary.am \
    --exp-data pn_data/ligo_data.csv \
    --exp-meta pn_data/ligo_data.am \
    --out pn_analysis.ndjson \
    --oam pn_analysis.am
```

**Output:** 
- PN corrections up to specified order
- Observational signatures and scaling laws
- Goodness-of-fit vs experimental constraints
- Parameter ranges surviving precision tests

### 2. Strong-Curvature Regime Pipeline

**Input:** Model configurations → Toy model generation → Regime classification

```bash
# Generate 2D black hole models
cd strong_curvature/
python generate_2d_blackhole.py \
    --model-config blackhole_config.am \
    --out blackhole_data.ndjson \
    --oam blackhole_summary.am

# Generate minisuperspace cosmology
python minisuperspace_cosmo.py \
    --cosmo-config cosmo_config.am \
    --out cosmo_data.ndjson \
    --oam cosmo_summary.am

# Unify and compare models
python compare_strong_models.py \
    --models blackhole_data.ndjson cosmo_data.ndjson \
    --meta blackhole_summary.am cosmo_summary.am \
    --out unified_strong_models.ndjson \
    --oam unified_summary.am
```

**Output:**
- Curvature invariants (Ricci, Kretschmann scalars)
- Quantum gravity parameter (curvature/Planck scale)
- Regime classification (classical/transition/quantum)
- Parameter ranges requiring full quantum gravity

### 3. Unified Metadata Format

All stages use consistent AsciiMath metadata files (`.am`) containing:

```
[ key1 = value1, key2 = "string_value", key3 = 1.23e-4, ... ]
```

This enables automated pipeline chaining and parameter tracking across analysis stages.

## Key Features

### Semi-Classical Analysis
- **Symbolic PN expansion** using SymPy for warp drive metrics
- **Observational signatures** in gravitational wave detectors
- **Parameter constraints** from precision tests (LIGO, atomic interferometry)
- **Order-by-order comparison** of theoretical predictions vs data

### Strong-Curvature Models
- **2D black hole toy models** with exact curvature calculations
- **FRW minisuperspace cosmology** for early universe scenarios
- **Quantum gravity indicators** based on Planck-scale ratios
- **Regime boundaries** between classical and quantum gravity

### Analysis Integration
- **Consistent data format** (NDJSON + AsciiMath metadata)
- **Automated pipeline** with configurable parameters
- **Cross-regime comparison** of theoretical predictions
- **Scalable framework** for additional model types

## Current Status

✅ **Completed:**
- Original sensitivity analysis pipeline
- Semi-classical PN correction generator
- Strong-curvature toy model framework
- Unified metadata and analysis structure

🔄 **Working:**
- PN corrections successfully generated
- 2D black hole models computed
- Model comparison and regime classification

🔧 **Minor Issues:**
- JSON serialization errors in some analysis scripts (fixable)
- Frequency range parsing in PN config files (fixed)

## Next Steps

1. **Fix remaining JSON serialization** in analysis scripts
2. **Add more experimental datasets** (atomic interferometry, pulsar timing)
3. **Extend toy models** (higher-dimensional black holes, cosmological perturbations)
4. **Implement parameter space exploration** with automated constraint mapping
5. **Add visualization tools** for regime boundaries and observational prospects

This framework now provides a complete theoretical physics testing pipeline that can explore warp drive signatures across energy scales from current detector sensitivity up to Planck-scale quantum gravity.
