import numpy as np
import pandas as pd
from scipy.stats import bootstrap

def compute_fi(df: pd.DataFrame, area_type: str) -> float:
    """
    Computes the Focality Index (FI) for neuronal activity across brain areas.
    FI = (Σ P_i²) / (Σ P_i)², where P_i is the proportion of significant neurons in area i.
    """
    if df.empty:
        return np.nan
        
    area_sums = df.groupby(area_type).agg(
        sig_neurons=('significant', 'sum'),
        total_neurons=('unit_id', 'count')
    )
    area_sums['P_i'] = area_sums['sig_neurons'] / area_sums['total_neurons']

    sum_of_squares = (area_sums['P_i'] ** 2).sum()
    square_of_sum = (area_sums['P_i'].sum()) ** 2
    
    return sum_of_squares / square_of_sum if square_of_sum > 0 else np.nan

# Bootstrapping Implementations

def hierarchical_bootstrap_fi(neuron_df: pd.DataFrame, area_type: str, n_boot: int = 10000, seed=None) -> tuple:
    """
    Performs a two-stage hierarchical bootstrap of the FI, resampling mice first,
    then neurons within the selected mice.
    """
    rng = np.random.default_rng(seed)
    mice = neuron_df['mouse_id'].unique()
    boots = np.empty(n_boot)
    
    for i in range(n_boot):
        # Stage 1: Resample mice with replacement
        sampled_mice_ids = rng.choice(mice, size=len(mice), replace=True)
        
        # Stage 2: Resample neurons within each originally sampled mouse group
        boot_df_parts = []
        for mouse_id in sampled_mice_ids:
            mouse_neurons = neuron_df[neuron_df['mouse_id'] == mouse_id]
            resampled_neurons = mouse_neurons.sample(n=len(mouse_neurons), replace=True, random_state=rng)
            boot_df_parts.append(resampled_neurons)
            
        boot_df = pd.concat(boot_df_parts, ignore_index=True)
        boots[i] = compute_fi(boot_df, area_type)
        
    return tuple(np.percentile(boots, [2.5, 97.5]))


def bca_bootstrap_fi(neuron_df: pd.DataFrame, area_type: str, n_boot: int = 10000) -> tuple:
    """
    Computes bias-corrected and accelerated (BCa) bootstrap CIs for the FI.
    """
    def fi_statistic_wrapper(indices):
        return compute_fi(neuron_df.iloc[indices], area_type)

    data = (np.arange(len(neuron_df)),)
    res = bootstrap(
        data,
        statistic=fi_statistic_wrapper,
        n_resamples=n_boot,
        method='bca',
        random_state=np.random.default_rng()
    )
    return res.confidence_interval.low, res.confidence_interval.high


def percentile_bootstrap_fi(neuron_df: pd.DataFrame, area_type: str, n_boot: int = 10000) -> tuple:
    """
    Computes percentile bootstrap CIs by resampling neurons with replacement.
    """
    fi_bootstrap_values = [
        compute_fi(neuron_df.sample(n=len(neuron_df), replace=True), area_type)
        for _ in range(n_boot)
    ]
    return tuple(np.percentile(fi_bootstrap_values, [2.5, 97.5]))
