# warp-sensitivity-analysis

Compare synthetic warp-bubble signals against a detector noise floor.

## Overview

This package provides two scripts to:
1. Generate a toy detector noise curve (frequency vs. strain-spectral-density).  
2. Analyze mock gravitational-wave signals against that noise curve and flag detectable modes.

## Repository Structure
```

.  
├── README.md  
├── mock\_data.ndjson # Synthetic signal time-series/spectra JSON-lines  
├── mock\_data.am # AsciiMath metadata for mock\_data  
├── sensitivity\_curve.csv # Generated noise curve (frequency, noise PSD)  
├── sensitivity\_curve.am # AsciiMath metadata for sensitivity curve  
├── generate\_sensitivity\_curve.py  
└── analyze\_sensitivity.py

```markdown
## Prerequisites

- Python 3.7+  
- NumPy  
- SciPy  
- ndjson  
- tqdm (optional, for progress bars)

Install via:

```bash
pip install numpy scipy python-ndjson tqdm
```

## Usage

### 1\. Generate Noise Curve

```bash
python generate_sensitivity_curve.py
```

This produces:

-   `sensitivity_curve.csv`
    
    -   Columns: `frequency_Hz`, `noise_strain_per_sqrtHz`
        
-   `sensitivity_curve.am`
    
    -   One-line AsciiMath metadata: model name, parameters, point count
        

### 2\. Analyze Signal Detectability

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
    

## Input/Output Formats

-   **`.ndjson`**
    
    -   One JSON object per line.
        
    -   For `mock_data.ndjson`: fields include  
        `{ label, time_series: [...], sampling_rate }`
        
-   **`.am`**
    
    -   Single-line AsciiMath metadata arrays:  
        `[ key1 = val1, key2 = val2, … ]`
        
-   **`.csv`**
    
    -   Standard comma-separated with header row.
        

## Examples

```bash
# 1. Build the noise curve
./generate_sensitivity_curve.py

# 2. Run detection analysis
./analyze_sensitivity.py \
  --mock mock_data.ndjson \
  --meta mock_data.am \
  --noise sensitivity_curve.csv \
  --nmeta sensitivity_curve.am \
  --out sensitivity_comparison.ndjson \
  --oam sensitivity_comparison.am

# 3. Inspect results
head sensitivity_comparison.ndjson
cat sensitivity_comparison.am
```