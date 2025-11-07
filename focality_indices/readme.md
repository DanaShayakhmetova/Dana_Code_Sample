## Project Structure

```
focality_indices/
├── config.py           # All configuration, paths, and constants.
├── data_processing.py  # Data loading, cleaning, and filtering.
├── statistics.py       # Core FI calculation and bootstrapping functions.
├── plotting.py         # Figure generation and saving.
└── main.py             # Main entry point to run the pipeline.
```

## Quick Start

### 1. Install Dependencies
```bash
pip install pandas numpy matplotlib scipy iblatlas openpyxl
```

### 2. Run the Pipeline
Execute the main script from your terminal. You can adjust parameters directly from the command line.

```bash
python main.py --bootstrap_method hierarchical --area_type swanson_region --thres 15
```

**Command-Line Arguments:**
*   `--bootstrap_method`: `percentile`, `bca`, or `hierarchical`.
*   `--area_type`: Brain area definition (`swanson_region`, etc.).
*   `--thres`: Minimum neuron count per area.
*   `--mouse_thres`: Minimum mouse count per area.
