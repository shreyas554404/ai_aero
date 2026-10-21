# ai_aero

AI-based predictive maintenance reference implementation for aircraft components.

## What this provides
- Multi-algorithm modeling per component:
  - Linear Regression
  - Decision Tree
  - Random Forest
  - Deep Learning (MLP)
  - XGBoost (or Gradient Boosting fallback when XGBoost is unavailable)
- Data input options:
  - synthetic data generation
  - CSV input (raw sensor datasets)
- Manufacturer/component-specific operating limits
- Per-component and per-flight airworthiness decisions
- 3D computational graphs per component plus a master combined 3D graph

## Run
```bash
python predictive_maintenance.py --source synthetic --output-dir outputs --n-samples 1200
```

For CSV input:
```bash
python predictive_maintenance.py --source csv --csv-path /absolute/path/to/sensors.csv --output-dir outputs
```

Expected CSV columns:
- `flight_id`
- `manufacturer`
- `component`
- `vibration`
- `temperature`
- `pressure`
- `cycles`
- `wear_index`

## Outputs
- `outputs/model_metrics.csv`
- `outputs/component_results.csv`
- `outputs/flight_airworthiness_summary.csv`
- `outputs/*_3d.png`
- `outputs/master_combined_3d.png`

## Limits and compliance
This repository includes configurable placeholder limits to support model development and demonstration. For operational or approval use, replace them with validated regulator/manufacturer-approved limits and verified legal compliance workflows.
