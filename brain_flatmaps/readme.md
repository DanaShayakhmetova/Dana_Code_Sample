## Folder Structure

```
brain_flatmaps/
├── config.py           # Configuration, paths, and constants.
├── data_processing.py  # Data loading, cleaning, and filtering.
├── plotting.py         # Visualization functions and figure generation.
├── plotting_utils.py   # Helper functions for saving figures and combining colormaps.
└── main.py             # Main entry point for running the pipeline.
```

## Quick Start
### 1. Install Dependencies
```bash
pip install pandas numpy matplotlib cmasher iblatlas
```

### 2. Run the Pipeline
Execute the main script from your terminal. You can modify parameters directly via command-line arguments.
```bash
python main.py --neuron_threshold 15 --mouse_threshold 3 --color_bar_type one_color_bar
```

**Command-Line Arguments**

```bash
--neuron_threshold: Minimum neurons per brain area.

--mouse_threshold: Minimum mice per brain area (set to 0 to disable).

--color_bar_type: Color bar style (one_color_bar, reward_group_colorbar, analysis_colorbar).
```


