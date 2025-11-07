import argparse
import config
import data_processing
import plotting

def main(args):
    """Main function to orchestrate the FI analysis pipeline."""
    print("Starting Focality Index pipeline...")

    # 1. Load and prepare data
    roc_df = data_processing.load_and_combine_data()
    roc_df = data_processing.prepare_area_columns(roc_df, args.area_type)
    
    # 2. Filter data based on thresholds
    filtered_df = data_processing.filter_by_neuron_count(
        roc_df, args.area_type, args.thres, args.mouse_thres
    )

    # 3. Generate all plots
    print("\nGenerating Focality Index figures...")
    
    # Plot by reward group for each single analysis
    for analysis in config.SINGLE_ANALYSES:
        print(f"  - Plotting by reward group for: {analysis}")
        plotting.plot_focality_index_by_reward_group(
            filtered_df, analysis, args.area_type, args.bootstrap_method
        )

    # Plot by analysis pair for each reward group
    for r_group in config.REWARD_GROUPS:
        for pair in config.PAIR_ANALYSES:
            print(f"  - Plotting by analysis for: {pair} ({r_group})")
            plotting.plot_focality_index_by_analysis(
                filtered_df, pair, r_group, args.area_type, args.bootstrap_method
            )

    print("\nPipeline finished successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute and plot Focality Indices from ROC results.")
    parser.add_argument(
        '--bootstrap_method', type=str, default='percentile',
        choices=['hierarchical', 'bca', 'percentile'],
        help='Bootstrap method for confidence intervals.'
    )
    parser.add_argument(
        '--area_type', type=str, default='swanson_region',
        choices=['swanson_region', 'area_custom', 'ccf_parent_acronym'],
        help='Brain area column to use for analysis.'
    )
    parser.add_argument('--thres', type=int, default=15, help='Minimum neuron count per area.')
    parser.add_argument('--mouse_thres', type=int, default=3, help='Minimum mouse count per area.')

    args = parser.parse_args()
    main(args)
