# Advance-ML

Exploratory analysis tooling for the door registry and maintenance datasets.

## Run the refactored pipeline

```bash
python scripts/run_eda_pipeline.py
```

Optional custom data path:

```bash
python scripts/run_eda_pipeline.py --data-dir src/datasets
```

## Generate the PowerPoint report

```bash
python scripts/build_report.py
```

The report is saved to `reports/Imbalanced_MC_Timeseries_Report.pptx`.

## Quick smoke test

```bash
python tests/smoke_test.py
```

## Project layout
- `src/eda_pipeline/` contains the refactored modules (loading, EDA, preprocessing, modeling).
- `scripts/run_eda_pipeline.py` is the entry point that reproduces the notebook workflow.
- `tests/smoke_test.py` is a lightweight import check.

## Notes
- Plots are displayed interactively; run from a notebook kernel or a Python environment with a display backend.
- The loader expects these filenames in the data directory:
  - `registry_no_missing (2).csv`
  - `catalog_numeric_missing (1).csv`
  - `history_numeric_missing (1).csv`
