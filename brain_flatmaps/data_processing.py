import glob
import os
import pandas as pd
import numpy as np
from iblatlas.atlas import BrainRegions
from typing import List

import config

def load_and_combine_data() -> pd.DataFrame:
    """Loads and combines ROC data from all mice and merges with metadata."""
    print("\nLoading and combining data...")
    # Load mouse metadata
    mouse_info_df = pd.read_excel(config.MOUSE_INFO_PATH)
    mouse_info_df.rename(columns={'mouse_name': 'mouse_id'}, inplace=True)
    valid_mice_filter = (
        (mouse_info_df['exclude'] == 0) &
        (mouse_info_df['exclude_ephys'] == 0) &
        (mouse_info_df['reward_group'].isin(['R+', 'R-'])) &
        (mouse_info_df['recording'] == 1)
    )
    mouse_info_df = mouse_info_df[valid_mice_filter]

    # Note: The suffixes "_ab" and "_mh" indicate which supervisor's mouse data is used (based on their initials).
    
    # Load ROC data from both sources
    data_path_ab = os.path.join(config.DATA_PATH, "TEMP_NAME_CSV")
    data_path_mh = os.path.join(config.DATA_PATH, "TEMP_NAME_CSV")

    files_ab = glob.glob(os.path.join(data_path_ab, '**', '*_roc_results_new.csv'), recursive=True)
    df_ab = pd.concat([pd.read_csv(f) for f in files_ab], ignore_index=True)

    files_mh = glob.glob(os.path.join(data_path_mh, '**', '*_roc_results_new.csv'), recursive=True)
    df_mh = pd.concat([pd.read_csv(f) for f in files_mh], ignore_index=True)

    roc_df = pd.concat([df_ab, df_mh], ignore_index=True)

    # Merge with mouse info and add a unique neuron ID
    roc_df = roc_df.merge(mouse_info_df[['mouse_id', 'reward_group']], on='mouse_id', how='left')
    roc_df['neuron_id'] = roc_df.index.astype(int)

    print(f"Data loaded for {roc_df['mouse_id'].nunique()} mice.")
    print("Done.")
    return roc_df


def swanson_conversion(roc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts CCF acronyms to Swanson regions and cleans the data.
    """
    br = BrainRegions()
    df = roc_df.copy()
    df['ccf_acronym'] = df['ccf_acronym'].astype(str).str.strip()

    # Manual mapping for specific missing acronyms
    manual_mapping = {
        'STR': 'STRv', 'HPF': 'CA1', 'OLF': 'AON', 'FRP6a': 'FRP',
        'FRP5': 'FRP', 'FRP4': 'FRP', 'FRP2/3': 'FRP', 'FRP1': 'FRP',
        'MB': 'MRN', 'P': 'PRNr', 'SSp-tr6a': 'SSp-tr', 'SSp-tr5': 'SSp-tr',
        'SSp-tr4': 'SSp-tr', 'SSp-tr2/3': 'SSp-tr', 'SSp-tr1': 'SSp-tr',
        'ORBm6a': 'ORBm', 'RSPagl6a': 'RSPd', 'RSPagl6b': 'RSPd',
        'RSPagl5': 'RSPd', 'RSPagl4': 'RSPd', 'RSPagl2/3': 'RSPd',
        'RSPagl1': 'RSPd',
    }
    df['ccf_acronym_mapped'] = df['ccf_acronym'].replace(manual_mapping)
    df.loc[df['ccf_acronym_mapped'].str.startswith('SSp-bfd', na=False), 'ccf_acronym_mapped'] = 'SSp-bfd'

    unique_acronyms = df['ccf_acronym_mapped'].dropna().unique()
    swanson_mapping = {
        acronym: (br.acronym2acronym(acronym, mapping='Swanson')[0] if br.acronym2acronym(acronym, mapping='Swanson').size > 0 else np.nan)
        for acronym in unique_acronyms if pd.notna(acronym)
    }
    df['swanson_region'] = df['ccf_acronym_mapped'].map(swanson_mapping)

    # Filter out unwanted regions
    regions_to_remove = ['root', 'void', '', 'nan', 'CTXsp', 'HY']
    df = df.dropna(subset=['swanson_region'])
    df = df[~df['swanson_region'].isin(regions_to_remove)]
    df = df.drop('ccf_acronym_mapped', axis=1)

    return df


def filter_number_of_neurons(df: pd.DataFrame, thres: int = 15, mouse_thres: int = 3) -> pd.DataFrame:
    """
    Filters out brain areas based on neuron and mouse count criteria.
    """
    # Exclude Pons and adjacent brainstem areas
    excluded_areas = {"PRNr", "PRNc", "RM", "PPN", "V", "PSV", "PG", "LAV", "NLL", "SUT"}
    df = df[~df['swanson_region'].isin(excluded_areas)]

    # Filter by neuron count
    counts = df.groupby(['reward_group', 'swanson_region'])['unit_id'].nunique().unstack(fill_value=0)
    areas_low_neurons = counts[(counts < thres).all(axis=1)].index.tolist()
    df = df[~df['swanson_region'].isin(areas_low_neurons)]
    print(f"Areas removed (count < {thres} neurons in both groups): {areas_low_neurons}")

    # Filter by mouse count
    if mouse_thres > 0:
        mouse_counts = df.groupby('swanson_region')['mouse_id'].nunique()
        areas_low_mice = mouse_counts[mouse_counts < mouse_thres].index.tolist()
        df = df[~df['swanson_region'].isin(areas_low_mice)]
        print(f"Areas removed (mice < {mouse_thres}): {areas_low_mice}")

    return df

def get_all_brain_region_names() -> np.ndarray:
    """Gets all unique Swanson atlas region acronyms from BrainRegions."""
    br = BrainRegions()
    swanson_indices = np.unique(br.mappings['Swanson'])
    swanson_acronyms = np.sort(br.acronym[swanson_indices])
    return swanson_acronyms
