import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb, Normalize
from typing import Tuple, List
import config
matplotlib.use('Agg') # Set backend to non-interactive

def save_flatmap_figure(fig: plt.Figure, filename: str, output_dir: str, formats: List[str] = None, dpi: int = 500):
    """Saves a figure in multiple specified formats."""
    if formats is None:
        formats = ['pdf', 'svg', 'png']
    os.makedirs(output_dir, exist_ok=True)
    for fmt in formats:
        filepath = os.path.join(output_dir, f"{filename}.{fmt}")
        fig.savefig(filepath, format=fmt, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        # print(f"Saved: {filepath}")

def get_analysis_colormap(analysis_type: str, intensity_range: Tuple[float, float] = (0.2, 1.0)) -> LinearSegmentedColormap:
    """Creates a custom colormap for a given analysis type."""
    key_map = {
        'whisker': 'whisker', 'auditory': 'auditory', 'spontaneous': 'spontaneous',
        'learning': 'learning', 'choice': 'choice', 'baseline': 'baseline_choice'
    }
    base_color_key = next((val for key, val in key_map.items() if key in analysis_type.lower()), 'base')
    base_color = config.PROJECT_COLORS.get(base_color_key, 'blues')

    # Create a gradient from white to the base color
    color_rgb = to_rgb(base_color)
    colors = [
        tuple( (1 - w) * 1.0 + w * c for c in color_rgb)
        for w in np.linspace(intensity_range[0], intensity_range[1], 256)
    ]
    return LinearSegmentedColormap.from_list(f'custom_{analysis_type}', colors)

def create_combined_colormap( cmap_base: LinearSegmentedColormap, vmin_metric: float, vmax_metric: float) -> Tuple[LinearSegmentedColormap, Normalize, float, float]:
    """Creates a colormap that includes a distinct color for missing data."""
    v_min_range = config.MISSING_DATA_DARK
    v_max_range = vmax_metric

    def normalize_val(v, v_min, v_max):
        return (v - v_min) / (v_max - v_min) if (v_max - v_min) != 0 else 0.5

    pos_dark = normalize_val(config.MISSING_DATA_DARK, v_min_range, v_max_range)
    pos_metric_start = normalize_val(vmin_metric, v_min_range, v_max_range)

    nodes = [(pos_dark, config.DARK_GRAY), (pos_dark + 1e-6, config.DARK_GRAY)]

    if pos_metric_start > pos_dark + 1e-6:
        start_color = cmap_base(0.0)
        nodes.extend([(pos_metric_start - 1e-6, start_color), (pos_metric_start, start_color)])

    if vmax_metric > vmin_metric:
        for m_node in np.linspace(0.0, 1.0, 256):
            metric_val = vmin_metric + m_node * (vmax_metric - vmin_metric)
            pos = normalize_val(metric_val, v_min_range, v_max_range)
            nodes.append((pos, cmap_base(m_node)))

    nodes.sort(key=lambda x: x[0])
    cmap_final = LinearSegmentedColormap.from_list('CustomGrayMetricMap', nodes)
    norm = Normalize(vmin=v_min_range, vmax=v_max_range)
    return cmap_final, norm, v_min_range, v_max_range
