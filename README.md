# warp-sensitivity-analysis

Multi-scale theoretical physics testing framework for warp drive signatures.

## Overview

This package provides a comprehensive analysis pipeline spanning three regimes:
1. **Signal vs. Noise Analysis:** Compare synthetic warp signals against detector sensitivity curves
2. **Semi-Classical Testing:** Generate Post-Newtonian corrections and compare with precision experiments  
3. **Strong-Curvature Models:** Explore Planck-scale physics with toy black hole and cosmological models

## Repository Structure
```
.  
├── README.md  
├── EXTENDED_FRAMEWORK.md           # Detailed framework documentation
├── mock_data.ndjson                # Synthetic signal time-series/spectra JSON-lines  
├── mock_data.am                    # AsciiMath metadata for mock_data  
├── sensitivity_curve.csv           # Generated noise curve (frequency, noise PSD)  
├── sensitivity_curve.am            # AsciiMath metadata for sensitivity curve  
├── generate_sensitivity_curve.py   # Noise curve generator
├── analyze_sensitivity.py          # Original signal vs noise analysis
├── 
├── semi_classical/                 # Post-Newtonian analysis
│   ├── compute_pn_corrections.py   # PN expansion generator
│   ├── analyze_pn_tests.py         # Experimental comparison
│   ├── theory_params.am            # Theory configuration
│   ├── pn_config.am                # PN expansion settings
│   └── pn_data/                    # Experimental datasets
│       ├── ligo_data.csv           # LIGO sensitivity curves
│       ├── ligo_data.am            # LIGO metadata
│       └── atomic_interf.am        # Atomic interferometry data
│
└── strong_curvature/               # Planck-scale models
    ├── generate_2d_blackhole.py    # 2D black hole toy models
    ├── minisuperspace_cosmo.py     # FRW minisuperspace cosmology
    ├── compare_strong_models.py    # Model unification
    ├── blackhole_config.am         # Black hole parameters
    └── cosmo_config.am             # Cosmology parameters
```markdown
## Prerequisites

- Python 3.7+  
- NumPy  
- SciPy
- SymPy (for symbolic PN calculations)
- ndjson  
- tqdm (optional, for progress bars)

Install via:

```bash
pip install numpy scipy sympy python-ndjson tqdm
```

## Usage

### 1. Generate Noise Curve

```bash
python generate_sensitivity_curve.py
```

This produces:

-   `sensitivity_curve.csv`
    
    -   Columns: `frequency_Hz`, `noise_strain_per_sqrtHz`
        
-   `sensitivity_curve.am`
    
    -   One-line AsciiMath metadata: model name, parameters, point count
        

### 2. Analyze Signal Detectability

```bash
python analyze_sensitivity.py `
  --mock   mock_data.ndjson `
  --meta   mock_data.am `
  --noise  sensitivity_curve.csv `
  --nmeta  sensitivity_curve.am `
  --out    sensitivity_comparison.ndjson `
  --oam    sensitivity_comparison.am
```

**Outputs:**

-   `sensitivity_comparison.ndjson`  
    JSON-line records:
    
```json
{
    "label": "signal1",
    "detectable": true,
    "snr": 12.5
}
```
    
-   `sensitivity_comparison.am`  
    AsciiMath summary: detector name, injection count, detection thresholds

### 3. Post-Newtonian Analysis

Generate PN corrections for warp drive theories:

```bash
cd semi_classical/
python compute_pn_corrections.py \
  --theory theory_params.am \
  --pn-config pn_config.am \
  --out pn_waveforms.ndjson \
  --oam pn_summary.am
```

**Configuration files:**
- `theory_params.am`: Contains `WarpVelocity`, `MassParameter`, `TheoryType`
- `pn_config.am`: Contains `MaxPNOrder`, `FrequencyRange`, `ExpansionParameter`

**Outputs:**
- `pn_waveforms.ndjson`: PN corrections with observational signatures
- `pn_summary.am`: Analysis metadata

Analyze against experimental data:

```bash
python analyze_pn_tests.py \
  --pn-data pn_waveforms.ndjson \
  --pn-meta pn_summary.am \
  --exp-data pn_data/ligo_data.csv \
  --exp-meta pn_data/ligo_data.am \
  --out pn_analysis.ndjson \
  --oam pn_analysis.am \
  --snr-threshold 5.0
```

**Flags and Input Specifications:**
- `--pn-data`: PN corrections from compute_pn_corrections.py (.ndjson format)
  - Each entry contains: `pn_order`, `correction` (symbolic expressions), `signature` (frequency-domain observational signatures), `v_over_c`, `mass_parameter`
- `--pn-meta`: PN metadata file (.am format) containing analysis parameters
  - Required keys: `max_pn_order`, `warp_velocity`, `frequency_range`, `n_corrections`
- `--exp-data`: Experimental sensitivity data (.csv with columns: frequency_hz, strain_sensitivity, experiment_type)
  - Compatible with LIGO, Virgo, atomic interferometry, and pulsar timing datasets
- `--exp-meta`: Experimental metadata (.am with keys: ExperimentType, InstrumentName, FrequencyRange, StrainSensitivity, DataSource)
- `--snr-threshold`: SNR threshold for detectability (default: 5.0, recommended range: 3.0-10.0)

**Outputs:**
- `pn_analysis.ndjson`: Detection analysis for each PN order with fields:
  - `pn_order`: PN correction order ("1PN", "2PN", "3PN", etc.)
  - `theory_params`: Contains v_over_c and mass_parameter values from theory
  - `experimental_analysis`: Per-experiment detectability results including SNR, parameter bounds, frequency ranges
  - `overall_detectability`: Boolean indicating if any experiment can detect this PN order
  - `best_snr`: Highest SNR achieved across all experimental datasets
- `pn_analysis.am`: Summary metadata including detection statistics, parameter constraints, and experiment coverage

### 4. Strong-Curvature Models

Generate 2D black hole curvature data:

```bash
cd strong_curvature/
python generate_2d_blackhole.py \
  --model-config blackhole_config.am \
  --out blackhole_data.ndjson \
  --oam blackhole_summary.am
```

**Configuration (`blackhole_config.am`):**
- `ModelType`: "2d_schwarzschild" or "warp_bubble"
- `MassParameter`: Characteristic mass scale
- `PlanckLength`: Planck length (default: 1e-35)
- `BubbleThickness`: For warp bubble models

**Outputs:**
- `blackhole_data.ndjson`: Curvature invariants and quantum parameters
- `blackhole_summary.am`: Model summary and regime classification

Generate minisuperspace cosmology:

```bash
python minisuperspace_cosmo.py \
  --cosmo-config cosmo_config.am \
  --out cosmo_data.ndjson \
  --oam cosmo_summary.am
```

Compare and unify strong-curvature models:

```bash
python compare_strong_models.py \
  --models blackhole_data.ndjson cosmo_data.ndjson \
  --meta blackhole_summary.am cosmo_summary.am \
  --out unified_strong_models.ndjson \
  --oam unified_summary.am
```

**Flags and Input Specifications:**
- `--models`: List of model data files (.ndjson) from toy model generators
  - Multiple files can be specified: `--models file1.ndjson file2.ndjson file3.ndjson`
  - Each .ndjson contains model results with curvature invariants, geodesic analysis, and Planck-scale physics indicators
  - Expected structure: `model_type`, `curvature_analysis`, `geodesic_analysis`, `planck_scale_analysis`
- `--meta`: Corresponding metadata files (.am) with model parameters and analysis settings
  - Must match order of --models files: `--meta meta1.am meta2.am meta3.am`
  - Required keys: `model_type`, analysis parameters specific to each model class

**Outputs:**
- `unified_strong_models.ndjson`: Unified comparison with regime classification, containing:
  - `model_id`: Unique identifier for each model in the comparison
  - `model_type`: Type of model ("2d_schwarzschild", "warp_bubble", "frw_minisuperspace", etc.)
  - `regime_classification`: Overall physical regime ("classical", "semi_classical", "quantum_gravity")
  - `curvature_analysis`: Extracted curvature scales and quantum gravity parameters
  - `classical_gr_valid`: Boolean indicating if classical General Relativity is adequate
  - `quantum_correction_strength`: Estimated strength of quantum gravity corrections (0.0-1.0)
  - `requires_quantum_gravity`: Boolean indicating if quantum effects dominate
  - `parameter_ranges`: Valid parameter ranges for classical and quantum regimes
- `unified_summary.am`: Summary statistics including:
  - `regime_distribution`: Count of models in each regime (classical/semi_classical/quantum_gravity)
  - `n_quantum_gravity`: Number of models requiring full quantum gravity treatment
  - `n_classical_valid`: Number of models where classical GR remains valid
  - `average_quantum_correction`: Average quantum correction strength across all models
  - `model_types_analyzed`: List of model types included in comparison
  - `planck_scale_physics_important`: Boolean indicating if any models reach Planck-scale physics
    

## Input/Output Formats

### AsciiMath Metadata Files (.am)

All configuration and metadata files use the AsciiMath format with consistent key-value pairs:

```
[ key1 = value1, key2 = "string_value", key3 = 1.23e-4, ... ]
```

**Configuration Keys by Script:**

**Semi-Classical Analysis:**
- `theory_params.am`:
  - `WarpVelocity`: Dimensionless warp velocity (v/c ratio)
  - `MassParameter`: Characteristic mass scale parameter
  - `TheoryType`: Theory model type ("alcubierre_warp", "van_den_broeck", etc.)
  - `MetricSignature`: Metric signature convention ("mostly_plus", "mostly_minus")

- `pn_config.am`:
  - `MaxPNOrder`: Maximum Post-Newtonian order to compute (integer: 1, 2, 3, ...)
  - `FrequencyRange`: Comma-separated frequency range in Hz ("10,1000")
  - `ExpansionParameter`: PN expansion parameter ("v_over_c", "frequency")
  - `SymbolicComputation`: Boolean for symbolic vs numerical computation

**Strong-Curvature Models:**
- `blackhole_config.am`:
  - `ModelType`: Model type ("2d_schwarzschild", "warp_bubble")
  - `MassParameter`: Characteristic mass scale in natural units
  - `PlanckLength`: Planck length in meters (default: 1e-35)
  - `BubbleThickness`: Warp bubble thickness parameter (for warp_bubble models)

- `cosmo_config.am`:
  - `ModelType`: Cosmology model ("frw_minisuperspace", "kasner")
  - `HubbleParameter`: Hubble parameter H₀ in units of 100 km/s/Mpc
  - `OmegaMatter`: Matter density parameter Ωₘ
  - `PlanckLength`: Planck length in meters
  - `ScaleFactorRange`: Comma-separated scale factor range ("1e-10,1e10")

**Experimental Data:**
- `ligo_data.am`, `atomic_interf.am`:
  - `ExperimentType`: Type of experiment ("gravitational_wave", "atomic_interferometry")
  - `InstrumentName`: Specific instrument name ("LIGO_H1", "LIGO_L1", "atom_interferometer")
  - `FrequencyRange`: Sensitive frequency range in Hz
  - `StrainSensitivity`: Characteristic strain sensitivity
  - `DataSource`: Source of experimental data

### JSON-Lines Data Files (.ndjson)

-   **One JSON object per line** for efficient streaming and processing
        
-   **For `mock_data.ndjson`**: 
    ```json
    { "label": "signal1", "time_series": [0.1, 0.2, ...], "sampling_rate": 16384 }
    ```
        
-   **For `pn_waveforms.ndjson`**:
    ```json
    { "pn_order": "1PN", "correction": {...}, "signature": {...}, "v_over_c": 0.1, "mass_parameter": 1e-6 }
    ```

-   **For `blackhole_data.ndjson`**:
    ```json
    { "model_type": "2d_schwarzschild", "curvature_analysis": {...}, "geodesic_analysis": {...} }
    ```

### CSV Data Files (.csv)

-   **Standard comma-separated with header row**
-   **For sensitivity curves**: `frequency_Hz, noise_strain_per_sqrtHz`
-   **For experimental data**: `frequency_hz, strain_sensitivity, experiment_type`
        

## Examples

```bash
# 1. Build the noise curve
python generate_sensitivity_curve.py

# 2. Run detection analysis
python analyze_sensitivity.py \
  --mock mock_data.ndjson \
  --meta mock_data.am \
  --noise sensitivity_curve.csv \
  --nmeta sensitivity_curve.am \
  --out sensitivity_comparison.ndjson \
  --oam sensitivity_comparison.am

# 3. Generate and analyze PN corrections
cd semi_classical/
python compute_pn_corrections.py \
  --theory theory_params.am \
  --pn-config pn_config.am \
  --out pn_waveforms.ndjson \
  --oam pn_summary.am

python analyze_pn_tests.py \
  --pn-data pn_waveforms.ndjson \
  --pn-meta pn_summary.am \
  --exp-data pn_data/ligo_data.csv \
  --exp-meta pn_data/ligo_data.am \
  --out pn_analysis.ndjson \
  --oam pn_analysis.am

# 4. Generate strong-curvature models
cd ../strong_curvature/
python generate_2d_blackhole.py \
  --model-config blackhole_config.am \
  --out blackhole_data.ndjson \
  --oam blackhole_summary.am

python compare_strong_models.py \
  --models blackhole_data.ndjson \
  --meta blackhole_summary.am \
  --out unified_models.ndjson \
  --oam unified_summary.am

# 5. Inspect results
head ../sensitivity_comparison.ndjson
cat ../sensitivity_comparison.am
head pn_analysis.ndjson
head unified_models.ndjson
```