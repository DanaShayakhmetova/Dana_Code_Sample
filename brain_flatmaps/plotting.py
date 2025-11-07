import os
import warnings
from typing import List, Tuple
import config
import data_processing
import plotting_utils
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from iblatlas.atlas import BrainRegions
from iblatlas.plots import plot_swanson_vector
from matplotlib.colors import ListedColormap, Normalize, TwoSlopeNorm

# Suppress expected warnings for empty slices and future changes
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='Mean of empty slice')


# Helper Functions
def generate_template_atlas(annotate: bool = True, hemisphere: str = 'both', figsize: tuple = (15, 10), dpi: int = 500) -> Tuple[plt.Figure, plt.Axes]:
    """Creates a blank Swanson flatmap template with optional region labels."""
    br = BrainRegions()
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    plot_swanson_vector(
        br=br,
        annotate=annotate,
        orientation='portrait',
        ax=ax,
        fontsize=8,
        hemisphere=hemisphere
    )
    ax.axis('off')
    return fig, ax


def get_data_and_regions(df_filtered: pd.DataFrame, metric: str, reward_group: str = None) -> Tuple[pd.Series, np.ndarray, mcolors.LinearSegmentedColormap, float, float, str]:
    """
    Computes the requested metric for each Swanson region.
    """
    df_mapped = data_processing.swanson_conversion(df_filtered)
    swanson_regions = df_mapped.swanson_region.unique()

    data = pd.Series(dtype=float)
    vmin_metric, vmax_metric = 0, 1
    label = 'Value'

    def compute_vlims(series: pd.Series, default: tuple = (0, 1)):
        series = series[series != config.MISSING_DATA_DARK].dropna()
        if series.empty:
            return default
        return round(series.min(), 1), round(series.max(), 1)

    total_counts = df_mapped.groupby('swanson_region').size()

    if metric == 'absolute':
        data = df_mapped['selectivity'].abs().groupby(df_mapped['swanson_region']).mean()
        label = 'Mean absolute selectivity'
        vmin_metric, vmax_metric = compute_vlims(data)
        vmin_metric, vmax_metric = 0, 1 # Standardize this metric

    elif metric == 'fraction':
        sig_counts = df_mapped[df_mapped['significant']].groupby('swanson_region').size()
        counts_df = pd.DataFrame({
            'total': total_counts,
            'selective': sig_counts.reindex(total_counts.index, fill_value=0)
        })
        data = (counts_df['selective'] / counts_df['total'] * 100).fillna(0)
        data[(counts_df['total'] > 0) & (counts_df['selective'] == 0)] = config.MISSING_DATA_DARK
        label = 'Selective units %'
        vmin_metric, vmax_metric = compute_vlims(data, default=(0, 100))

    else:
        raise ValueError(f"Unknown metric: {metric}")

    analysis_type = df_filtered['analysis_type'].iloc[0] if not df_filtered.empty else 'base'
    cmap = plotting_utils.get_analysis_colormap(f"{analysis_type}_{reward_group}" if reward_group else analysis_type)

    return data, swanson_regions, cmap, vmin_metric, vmax_metric, label


#Single Hemisphere Plotting

def generate_single_hemisphere(
    df: pd.DataFrame,
    analysis_type: str,
    reward_group: str,
    metric: str = 'absolute',
    annotate: bool = True,
    figsize: tuple = (10, 12)
) -> Tuple[plt.Figure, plt.Axes, str]:
    """
    Generates a single hemisphere flatmap for one analysis and reward group.
    """
    br = BrainRegions()
    all_swanson_regions = data_processing.get_all_brain_region_names()

    df_analysis = df[(df['analysis_type'] == analysis_type) & (df['reward_group'] == reward_group)].copy()
    data, _, cmap_base, vmin_metric, vmax_metric, label = get_data_and_regions(df_analysis, metric, reward_group)

    data_vector = data.reindex(br.acronym)
    cmap_final, norm, _, _ = plotting_utils.create_combined_colormap(cmap_base, vmin_metric, vmax_metric)
    data_to_plot = data_vector.dropna()
    region_ids_to_plot = br.acronym2id(data_to_plot.index.values)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_swanson_vector(
        region_ids_to_plot, data_to_plot.values,
        cmap=cmap_final, norm=norm, orientation='portrait', br=br, ax=ax,
        empty_color=config.LIGHT_GRAY, annotate=annotate,
        annotate_list=all_swanson_regions if annotate else []
    )

    cbar_ax = fig.add_axes([0.3, 0.95, 0.15, 0.01])
    sm = cm.ScalarMappable(cmap=cmap_base, norm=Normalize(vmin=vmin_metric, vmax=vmax_metric))
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(label, fontsize=7)
    cbar.ax.tick_params(labelsize=7)
    cbar.ax.xaxis.set_label_position('top')

    ax.axis('off')
    plt.tight_layout()
    filename = f"{analysis_type}_{reward_group}_{metric}_{'annotated' if annotate else 'not_annotated'}"
    return fig, ax, filename


# Dual Hemisphere Plotting

def _plot_dual_with_single_colorbar(fig, ax, br, left_data, right_data, cmap, vmin, vmax, label, left_title, right_title, annotate):
    """Helper to plot dual hemispheres with one shared colorbar."""
    all_swanson_regions = data_processing.get_all_brain_region_names()
    left_vector = left_data.reindex(all_swanson_regions)
    right_vector = right_data.reindex(all_swanson_regions)
    cmap_final, norm, _, _ = plotting_utils.create_combined_colormap(cmap, vmin, vmax)

    left_ids = -br.acronym2id(left_vector.dropna().index)
    right_ids = br.acronym2id(right_vector.dropna().index)
    combined_regions = np.concatenate([left_ids, right_ids])
    combined_values = np.concatenate([left_vector.dropna().values, right_vector.dropna().values])

    plot_swanson_vector(
        combined_regions, combined_values, hemisphere='both', cmap=cmap_final, norm=norm,
        orientation='portrait', br=br, ax=ax, annotate=annotate,
        annotate_list=all_swanson_regions if annotate else [], empty_color=config.LIGHT_GRAY
    )

    sm = cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    cbar_ax = fig.add_axes([0.42, 0.97, 0.16, 0.008])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(label, fontsize=8)
    cbar.ax.xaxis.set_label_position('top')

    ax.text(0.03, 1, left_title, transform=ax.transAxes, fontsize=8, va='top', bbox=dict(facecolor='white', alpha=0.8))
    ax.text(0.97, 1, right_title, transform=ax.transAxes, fontsize=8, va='top', ha='right', bbox=dict(facecolor='white', alpha=0.8))


def _plot_dual_with_two_colorbars(fig, ax, br, left_data, right_data, cmaps, vlims, label, titles, annotate):
    """
    Helper to plot dual hemispheres with two separate colorbars by creating a
    stitched colormap and offsetting the right hemisphere's data.
    """
    cmap_left, cmap_right = cmaps['left'], cmaps['right']
    vmin_left, vmax_left = vlims['left']
    vmin_right, vmax_right = vlims['right']
    left_title, right_title = titles['left'], titles['right']

    vmin_shared = min(vmin_left, vmin_right)
    vmax_shared = max(vmax_left, vmax_right)
    norm_on_shared = mcolors.Normalize(vmin=vmin_shared, vmax=vmax_shared)

    offset = vmax_shared + 10

    all_swanson_regions = data_processing.get_all_brain_region_names()
    left_vector = left_data.reindex(all_swanson_regions)
    right_vector = right_data.reindex(all_swanson_regions)

    right_data_offset = right_vector.copy()
    mask_to_offset = right_data_offset.notna() & (right_data_offset != config.MISSING_DATA_DARK)
    right_data_offset[mask_to_offset] += offset

    combined_regions = np.concatenate([-br.acronym2id(left_vector.dropna().index), br.acronym2id(right_data_offset.dropna().index)])
    combined_values = np.concatenate([left_vector.dropna().values, right_data_offset.dropna().values])

    vmin_norm = config.MISSING_DATA_DARK
    vmax_norm = offset + vmax_shared
    norm_temp = mcolors.Normalize(vmin=vmin_norm, vmax=vmax_norm)

    nodes = [(norm_temp(config.MISSING_DATA_DARK), config.DARK_GRAY)]
    for val in np.linspace(vmin_shared, vmax_shared, 128):
        nodes.append((norm_temp(val), cmap_left(norm_on_shared(val))))
    gap_start, gap_end = vmax_shared + 1e-6, offset + vmin_shared - 1e-6
    nodes.append((norm_temp(gap_start), config.LIGHT_GRAY))
    nodes.append((norm_temp(gap_end), config.LIGHT_GRAY))
    for val in np.linspace(vmin_shared, vmax_shared, 128):
        nodes.append((norm_temp(offset + val), cmap_right(norm_on_shared(val))))
    nodes.sort(key=lambda x: x[0])
    cmap_final = mcolors.LinearSegmentedColormap.from_list("stitched_cmap", nodes)

    plot_swanson_vector(
        combined_regions, combined_values, hemisphere='both', cmap=cmap_final,
        vmin=vmin_norm, vmax=vmax_norm, orientation='portrait', br=br, ax=ax,
        annotate=annotate, annotate_list=all_swanson_regions if annotate else [],
        empty_color=config.LIGHT_GRAY
    )

    norm_cbar_shared = Normalize(vmin=vmin_shared, vmax=vmax_shared)
    sm_left = cm.ScalarMappable(cmap=cmap_left, norm=norm_cbar_shared)
    cbar_ax_left = fig.add_axes([0.2, 0.97, 0.2, 0.01])
    cbar_left = fig.colorbar(sm_left, cax=cbar_ax_left, orientation='horizontal')
    cbar_left.set_label(label, fontsize=8)
    cbar_left.ax.tick_params(labelsize=8)

    sm_right = cm.ScalarMappable(cmap=cmap_right, norm=norm_cbar_shared)
    cbar_ax_right = fig.add_axes([0.6, 0.97, 0.2, 0.01])
    cbar_right = fig.colorbar(sm_right, cax=cbar_ax_right, orientation='horizontal')
    cbar_right.set_label(label, fontsize=8)
    cbar_right.ax.tick_params(labelsize=8)

    ax.text(0.03, 0.97, left_title, transform=ax.transAxes, fontsize=8, va='top', bbox=dict(facecolor='white', alpha=0.8))
    ax.text(0.97, 0.97, right_title, transform=ax.transAxes, fontsize=8, va='top', ha='right', bbox=dict(facecolor='white', alpha=0.8))


def generate_dual_hemispheres(
    df: pd.DataFrame,
    analysis_types: List[str],
    color_bar_type: str,
    reward_group: str = None,
    metric: str = 'absolute',
    annotate: bool = True,
    figsize: tuple = (15, 10)
) -> Tuple[plt.Figure, plt.Axes, str]:
    """Generates a dual hemisphere comparison flatmap."""
    br = BrainRegions()
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.03)

    # Mode 1: Compare two different analyses
    if len(analysis_types) == 2:
        if reward_group is None:
            raise ValueError("A reward group must be specified for a two-analysis comparison.")
        a1, a2 = analysis_types
        df1 = df[(df['analysis_type'] == a1) & (df['reward_group'] == reward_group)]
        df2 = df[(df['analysis_type'] == a2) & (df['reward_group'] == reward_group)]
        left_data, _, cmap_left, vmin_left, vmax_left, label = get_data_and_regions(df1, metric, reward_group)
        right_data, _, cmap_right, vmin_right, vmax_right, _ = get_data_and_regions(df2, metric, reward_group)
        
        vmin = min(vmin_left, vmin_right)
        vmax = max(vmax_left, vmax_right)
        left_title = f'{a1.replace("_", " ").title()} {reward_group}'
        right_title = f'{a2.replace("_", " ").title()} {reward_group}'
        filename = f"{a1}_vs_{a2}_{reward_group}_{metric}"
        
        cmap = plotting_utils.get_analysis_colormap(a1)
        _plot_dual_with_single_colorbar(fig, ax, br, left_data, right_data, cmap, vmin, vmax, label, left_title, right_title, annotate)

    # Mode 2: Compare R+ vs R- for one analysis
    elif len(analysis_types) == 1 and reward_group is None and metric != 'sign':
        analysis = analysis_types[0]
        df_r_plus = df[(df['analysis_type'] == analysis) & (df['reward_group'] == 'R+')]
        df_r_minus = df[(df['analysis_type'] == analysis) & (df['reward_group'] == 'R-')]
        left_data, _, cmap_left, vmin_left, vmax_left, label = get_data_and_regions(df_r_plus, metric, 'R+')
        right_data, _, cmap_right, vmin_right, vmax_right, _ = get_data_and_regions(df_r_minus, metric, 'R-')
        left_title = f'{analysis.replace("_", " ").title()} R+'
        right_title = f'{analysis.replace("_", " ").title()} R-'
        filename = f"{analysis}_R+_vs_R-_{metric}"

        if color_bar_type == 'reward_group_colorbar':
             _plot_dual_with_two_colorbars(
                fig, ax, br, left_data, right_data,
                cmaps={'left': cmap_left, 'right': cmap_right},
                vlims={'left': (vmin_left, vmax_left), 'right': (vmin_right, vmax_right)},
                label=label, titles={'left': left_title, 'right': right_title}, annotate=annotate
            )
        else:
            vmin = min(vmin_left, vmin_right)
            vmax = max(vmax_left, vmax_right)
            _plot_dual_with_single_colorbar(fig, ax, br, left_data, right_data, cmap_left, vmin, vmax, label, left_title, right_title, annotate)

    else:
        plt.close(fig)
        raise ValueError("Invalid parameters for dual hemisphere plot generation.")

    ax.axis('off')
    final_filename = f"{filename}_{'annotated' if annotate else 'not_annotated'}"
    return fig, ax, final_filename


# Delta Plotting

def plot_dual_hemispheres_delta(df: pd.DataFrame, analysis_type_1: str, analysis_type_2: str, metric: str, annotate: bool = True, figsize: tuple = (10, 12)):
    """Plots the change (delta) between two analyses for both reward groups."""
    br = BrainRegions()
    df_swanson = data_processing.swanson_conversion(df)

    def calculate_delta(data, a1, a2, rg, met):
        f1 = data[(data["analysis_type"] == a1) & (data["reward_group"] == rg)]
        f2 = data[(data["analysis_type"] == a2) & (data["reward_group"] == rg)]
        if met == 'fraction':
            m1 = f1.groupby("swanson_region")["significant"].apply(lambda x: 100 * x.sum() / len(x) if len(x) > 0 else 0)
            m2 = f2.groupby("swanson_region")["significant"].apply(lambda x: 100 * x.sum() / len(x) if len(x) > 0 else 0)
        else:
            m1 = f1.groupby("swanson_region")["selectivity"].apply(lambda x: x.abs().mean())
            m2 = f2.groupby("swanson_region")["selectivity"].apply(lambda x: x.abs().mean())
        return (m2.reindex(m1.index, fill_value=0) - m1).fillna(0)

    delta_rplus = calculate_delta(df_swanson, analysis_type_1, analysis_type_2, "R+", metric)
    delta_rminus = calculate_delta(df_swanson, analysis_type_1, analysis_type_2, "R-", metric)

    all_regions = data_processing.get_all_brain_region_names()
    left_vector = delta_rplus.reindex(all_regions)
    right_vector = delta_rminus.reindex(all_regions)

    vmin, vmax = (-0.1, 0.1) if metric == 'absolute' else (-15, 15)
    cmap = plt.get_cmap('PRGn')
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    combined_regions = np.concatenate([-br.acronym2id(left_vector.dropna().index), br.acronym2id(right_vector.dropna().index)])
    combined_values = np.concatenate([left_vector.dropna().values, right_vector.dropna().values])

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_swanson_vector(
        combined_regions, combined_values, hemisphere='both', cmap=cmap, norm=norm,
        orientation='portrait', br=br, ax=ax, annotate=annotate,
        annotate_list=all_regions if annotate else [], empty_color=config.LIGHT_GRAY
    )

    cbar_ax = fig.add_axes([0.4, 0.95, 0.2, 0.02])
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(f'Delta {metric} selectivity', fontsize=8)
    ax.axis('off')

    filename = f"{analysis_type_1}_to_{analysis_type_2}_delta_{metric}"
    return fig, ax, filename


# Generate all plots needed for project

def generate_all_plots(df: pd.DataFrame, color_bar_type: str):
    """
    Main function to generate all required flatmaps.
    """
    print("\n--- Generating All Flatmaps ---")

    # Single Hemisphere Flatmaps
    print("\n1. Generating Single Hemisphere Flatmaps...")
    output_dir = os.path.join(config.FIGURE_PATH, "single_hemisphere")
    for analysis in config.SINGLE_ANALYSES:
        for reward_group in config.REWARD_GROUPS:
            for metric in ['absolute', 'fraction']:
                for annotate in [True, False]:
                    try:
                        fig, _, filename = generate_single_hemisphere(df, analysis, reward_group, metric, annotate)
                        plotting_utils.save_flatmap_figure(fig, filename, output_dir)
                        plt.close(fig)
                    except Exception as e:
                        print(f"ERROR: {analysis}, {reward_group}, {metric}: {e}")

    # Dual Hemisphere Flatmaps
    print("\n2. Generating Dual Hemisphere Flatmaps...")
    output_dir = os.path.join(config.FIGURE_PATH, "dual_hemisphere")
    # Compare two analyses
    for pair in config.PAIR_ANALYSES:
        for reward_group in config.REWARD_GROUPS:
             for metric in ['absolute', 'fraction']:
                try:
                    fig, _, filename = generate_dual_hemispheres(df, pair, color_bar_type, reward_group=reward_group, metric=metric)
                    plotting_utils.save_flatmap_figure(fig, filename, output_dir)
                    plt.close(fig)
                except Exception as e:
                    print(f"ERROR: {pair}, {reward_group}: {e}")
    # Compare R+ vs R-
    for analysis in config.SINGLE_ANALYSES:
         for metric in ['absolute', 'fraction']:
            try:
                fig, _, filename = generate_dual_hemispheres(df, [analysis], color_bar_type, metric=metric)
                plotting_utils.save_flatmap_figure(fig, filename, output_dir)
                plt.close(fig)
            except Exception as e:
                print(f"ERROR: R+/- for {analysis}: {e}")

    # Delta Flatmaps
    print("\n3. Generating Delta Flatmaps...")
    output_dir = os.path.join(config.FIGURE_PATH, "delta_flatmaps")
    for pair in config.DELTA_PAIRS:
        for metric in ['absolute', 'fraction']:
            try:
                fig, _, filename = plot_dual_hemispheres_delta(df, pair[0], pair[1], metric)
                metric_dir = os.path.join(output_dir, metric)
                plotting_utils.save_flatmap_figure(fig, filename, metric_dir)
                plt.close(fig)
            except Exception as e:
                print(f"ERROR: delta map for {pair}: {e}")

    print("\n--- All plotting complete ---")
