import os
import config
import statistics
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


# Set non-interactive backend for matplotlib
matplotlib.use('Agg')

def plot_focality_index_by_reward_group(roc: pd.DataFrame, analysis_type: str, area_type: str, bootstrap_method: str):
    """Computes and plots per-group FI with bootstrapped CIs."""
    save_dir = os.path.join(config.FIGURE_PATH, area_type, bootstrap_method, "by_reward_group")
    os.makedirs(save_dir, exist_ok=True)

    roc_analysis = roc[roc.analysis_type == analysis_type]
    fi_vals, ci_vals = [], []

    for group in config.REWARD_GROUPS:
        roc_group = roc_analysis[roc_analysis.reward_group == group]
        fi_val = statistics.compute_fi(roc_group, area_type)
        
        if bootstrap_method == 'hierarchical':
            ci = statistics.hierarchical_bootstrap_fi(roc_group, area_type)
        elif bootstrap_method == 'bca':
            ci = statistics.bca_bootstrap_fi(roc_group, area_type)
        else: # Default to percentile
            ci = statistics.percentile_bootstrap_fi(roc_group, area_type)
        
        fi_vals.append(fi_val)
        ci_vals.append(ci)
    
    _create_fi_plot(
        x_labels=config.REWARD_GROUPS,
        fi_vals=fi_vals,
        ci_vals=ci_vals,
        n_areas=roc_analysis[area_type].nunique(),
        title=f"{analysis_type.replace('_', ' ').capitalize()}",
        save_path=os.path.join(save_dir, f"fi_{analysis_type}.png")
    )

def plot_focality_index_by_analysis(roc: pd.DataFrame, analyses: list, reward_group: str, area_type: str, bootstrap_method: str):
    """Computes and plots FI across multiple analyses for a given reward group."""
    save_dir = os.path.join(config.FIGURE_PATH, area_type, bootstrap_method, "by_analysis")
    os.makedirs(save_dir, exist_ok=True)
    
    roc_group = roc[(roc.reward_group == reward_group) & (roc.analysis_type.isin(analyses))]
    fi_vals, ci_vals = [], []

    for analysis in analyses:
        roc_analysis = roc_group[roc_group.analysis_type == analysis]
        fi_val = statistics.compute_fi(roc_analysis, area_type)

        if bootstrap_method == 'hierarchical':
            ci = statistics.hierarchical_bootstrap_fi(roc_analysis, area_type)
        elif bootstrap_method == 'bca':
            ci = statistics.bca_bootstrap_fi(roc_analysis, area_type)
        else: # Default to percentile
            ci = statistics.percentile_bootstrap_fi(roc_analysis, area_type)

        fi_vals.append(fi_val)
        ci_vals.append(ci)
    
    title = f"{analyses[0].replace('_', ' ')} vs {analyses[1].replace('_', ' ')} ({reward_group})"
    filename = f"fi_{analyses[0]}_vs_{analyses[1]}_{reward_group}.png"
    _create_fi_plot(
        x_labels=analyses,
        fi_vals=fi_vals,
        ci_vals=ci_vals,
        n_areas=roc_group[area_type].nunique(),
        title=title,
        save_path=os.path.join(save_dir, filename),
        rotate_labels=True
    )


def _create_fi_plot(x_labels, fi_vals, ci_vals, n_areas, title, save_path, rotate_labels=False):
    """Generic helper function to create an FI plot."""
    fi_errors = np.abs([[m - ci[0], ci[1] - m] for m, ci in zip(fi_vals, ci_vals)]).T
    uniform_fi = 1 / n_areas if n_areas > 0 else 0

    plt.figure(figsize=(3.5, 5))
    ax = plt.gca()
    ax.errorbar(x_labels, fi_vals, yerr=fi_errors, fmt='o', color='black', capsize=5, linestyle='none')
    ax.axhline(y=uniform_fi, color='gray', linestyle='--', label=f'Uniform (1/N, N={n_areas})')

    ax.set_ylabel('Focality Index')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(-0.7, len(x_labels) - 0.3)
    ax.set_ylim(bottom=0, top=max(0.06, uniform_fi * 2, np.nanmax([c[1] for c in ci_vals]) * 1.2))
    ax.set_title(title, fontsize=10)
    if rotate_labels:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.close()
