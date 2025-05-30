# Warp Sensitivity Analysis - Example Data

This directory contains sample data files and configurations for testing the complete analysis pipeline.

## Quick Start Examples

### 1. Basic Signal vs Noise Analysis
```bash
# From main directory
python generate_sensitivity_curve.py
python analyze_sensitivity.py \
  --mock examples/example_signals.ndjson \
  --meta examples/example_signals.am \
  --noise sensitivity_curve.csv \
  --nmeta sensitivity_curve.am \
  --out examples/detection_results.ndjson \
  --oam examples/detection_results.am
```

### 2. Post-Newtonian Analysis Pipeline
```bash
cd semi_classical/
python compute_pn_corrections.py \
  --theory theory_params.am \
  --pn-config pn_config.am \
  --out ../examples/pn_waveforms_example.ndjson \
  --oam ../examples/pn_summary_example.am

python analyze_pn_tests.py \
  --pn-data ../examples/pn_waveforms_example.ndjson \
  --pn-meta ../examples/pn_summary_example.am \
  --exp-data pn_data/ligo_data.csv \
  --exp-meta pn_data/ligo_data.am \
  --out ../examples/pn_analysis_example.ndjson \
  --oam ../examples/pn_analysis_example.am
```

### 3. Strong Curvature Models
```bash
cd strong_curvature/
python generate_2d_blackhole.py \
  --model-config blackhole_config.am \
  --out ../examples/blackhole_example.ndjson \
  --oam ../examples/blackhole_example.am

python compare_strong_models.py \
  --models ../examples/blackhole_example.ndjson \
  --meta ../examples/blackhole_example.am \
  --out ../examples/unified_example.ndjson \
  --oam ../examples/unified_example.am
```

## File Descriptions

- `example_signals.*`: Sample warp drive signals with different parameters
- `*_example.*`: Output files from running the example pipelines
- `test_config_*.am`: Alternative configuration files for testing different scenarios
