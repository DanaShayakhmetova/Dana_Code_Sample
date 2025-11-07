import argparse
import warnings

import config
import data_processing
import flatmap_generator

# Suppress nan slice warning, as it's expected
warnings.filterwarnings('ignore')

def main(args):
    """Main pipeline to load data, filter, and generate all flatmaps."""
    print("Starting flatmap generation pipeline...")

    # 1. Load and process data
    roc_df = data_processing.load_and_combine_data()
    roc_df_swanson = data_processing.swanson_conversion(roc_df)

    # 2. Filter data based on neuron and mouse counts
    print(f"\nFiltering areas with at least {args.neuron_threshold} neurons and {args.mouse_threshold} mice.")
    filtered_df = data_processing.filter_number_of_neurons(
        roc_df_swanson,
        thres=args.neuron_threshold,
        mouse_thres=args.mouse_threshold
    )
    print("Data filtering complete.")

    # 3. Generate all flatmaps
    print("\nGenerating single and dual hemisphere flatmaps...")
    flatmap_generator.generate_all_flatmaps(
        filtered_df,
        args.color_bar_type,
        config.FIGURE_PATH
    )
    print("Done.")

    # 4. Generate delta flatmaps
    print("\nGenerating delta flatmaps across conditions...")
    delta_abs_path = os.path.join(config.FIGURE_PATH, "delta_flatmaps", "mean_absolute")
    delta_frac_path = os.path.join(config.FIGURE_PATH, "delta_flatmaps", "fraction")

    for pair in config.DELTA_PAIRS:
        flatmap_generator.plot_dual_hemispheres_delta(
            filtered_df, pair[0], pair[1], 'absolute', delta_abs_path
        )
        flatmap_generator.plot_dual_hemispheres_delta(
            filtered_df, pair[0], pair[1], 'fraction', delta_frac_path
        )
    print("Delta flatmaps generated.")

    print("\nPipeline finished successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate brain flatmaps from ROC analysis results.")

    parser.add_argument(
        '--neuron_threshold',
        type=int,
        default=15,
        help='Minimum number of neurons to include a brain area.'
    )
    parser.add_argument(
        '--mouse_threshold',
        type=int,
        default=3,
        help='Minimum number of mice to include a brain area (set to 0 to disable).'
    )
    parser.add_argument(
        '--color_bar_type',
        type=str,
        default='one_color_bar',
        choices=['one_color_bar', 'reward_group_colorbar', 'analysis_colorbar'],
        help='Controls the color bar display style for dual hemisphere plots.'
    )

    args = parser.parse_args()
    main(args)
