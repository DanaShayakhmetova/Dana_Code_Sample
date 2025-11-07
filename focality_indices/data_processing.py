import os
import glob
import config
import pandas as pd
from iblatlas.atlas import BrainRegions

# Note: external helper file created by supervisor
import allen_utils_old as allen


def load_and_combine_data() -> pd.DataFrame:
    """Loads and combines ROC data from all mice and merges with metadata."""
    print("\nLoading and combining data...")

    mouse_info_df = pd.read_excel(config.MOUSE_INFO_PATH)
    mouse_info_df.rename(columns={'mouse_name': 'mouse_id'}, inplace=True)
    valid_mice_filter = (
        (mouse_info_df['exclude'] == 0) &
        (mouse_info_df['reward_group'].isin(['R+', 'R-'])) &
        (mouse_info_df['recording'] == 1)
    )
    mouse_info_df = mouse_info_df[valid_mice_filter]

    # Note: The suffixes "_ab" and "_mh" indicate which supervisor's mouse data is used (based on their initials).

    data_path_ab = os.path.join(config.DATA_PATH, "new_roc_csv_AB")
    data_path_mh = os.path.join(config.DATA_PATH, "new_roc_csv_MH")

    df_ab = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(data_path_ab, '**', '*.csv'), recursive=True)], ignore_index=True)
    df_mh = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(data_path_mh, '**', '*.csv'), recursive=True)], ignore_index=True)

    roc_df = pd.concat([df_ab, df_mh], ignore_index=True)
    roc_df = roc_df.merge(mouse_info_df[['mouse_id', 'reward_group']], on='mouse_id', how='left')
    roc_df['unit_id'] = roc_df.index.astype(int) # Use a more robust unique ID

    print(f"Data loaded for {roc_df['mouse_id'].nunique()} mice.")
    return roc_df


def prepare_area_columns(df: pd.DataFrame, area_type: str) -> pd.DataFrame:
    """Creates the specified brain area column for analysis."""
    print(f"\nPreparing area column: {area_type}...")
    if area_type == "swanson_region":
        return swanson_conversion(df)
    elif area_type == "area_custom":
        if allen:
            return allen.create_area_custom_column(df)
        else:
            raise RuntimeError("'allen_utils_old' is required for 'area_custom' but could not be imported.")
    return df # Defaults to using 'ccf_parent_acronym'


def swanson_conversion(roc_df: pd.DataFrame) -> pd.DataFrame:
    """Maps CCF acronyms to Swanson regions."""
    br = BrainRegions()
    df = roc_df.copy()
    df['ccf_acronym'] = df['ccf_acronym'].astype(str).str.strip()
    
    # Manual mapping logic 
    manual_mapping_dict = {
        'STR': 'STRv', 'HPF': 'CA1', 'OLF': 'AON', 'FRP6a': 'FRP', 'FRP5': 'FRP',
        'FRP4': 'FRP', 'FRP2/3': 'FRP', 'FRP1': 'FRP', 'MB': 'MRN', 'P': 'PRNr',
        'SSp-tr6a': 'SSp-tr', 'SSp-tr5': 'SSp-tr', 'SSp-tr4': 'SSp-tr',
        'SSp-tr2/3': 'SSp-tr', 'SSp-tr1': 'SSp-tr', 'ORBm6a': 'ORBm',
        'RSPagl6a': 'RSPd', 'RSPagl6b': 'RSPd', 'RSPagl5': 'RSPd',
        'RSPagl4': 'RSPd', 'RSPagl2/3': 'RSPd', 'RSPagl1': 'RSPd',
    }
    df['ccf_acronym_mapped'] = df['ccf_acronym'].replace(manual_mapping_dict)
    ssp_bfd_mask = df['ccf_acronym_mapped'].str.startswith('SSp-bfd', na=False)
    df.loc[ssp_bfd_mask, 'ccf_acronym_mapped'] = 'SSp-bfd'

    unique_acronyms = df['ccf_acronym_mapped'].dropna().unique()
    swanson_mapping = {
        acronym: (br.acronym2acronym(acronym, mapping='Swanson')[0] if br.acronym2acronym(acronym, mapping='Swanson').size > 0 else pd.NA)
        for acronym in unique_acronyms
    }
    df['swanson_region'] = df['ccf_acronym_mapped'].map(swanson_mapping)

    regions_to_remove = ['root', 'void', '', 'CTXsp', 'nan', 'HY']
    df = df.dropna(subset=['swanson_region'])
    df = df[~df['swanson_region'].isin(regions_to_remove)]
    return df.drop(columns=['ccf_acronym_mapped'])


def filter_by_neuron_count(df: pd.DataFrame, area_type: str, thres: int, mouse_thres: int) -> pd.DataFrame:
    """Filters brain areas based on neuron and mouse count criteria."""
    print(f"\nFiltering data with neuron threshold={thres} and mouse threshold={mouse_thres}...")
    
    # Exclude pons and adjacent brainstem areas
    excluded_areas = {"PRNr", "PRNc", "RM", "PPN", "V", "PSV", "PG", "LAV", "NLL", "SUT"}
    df = df[~df[area_type].isin(excluded_areas)]

    # Filter by neuron count across reward groups
    counts = df.groupby(['reward_group', area_type])['unit_id'].nunique().unstack(fill_value=0)
    low_neuron_areas = counts[(counts < thres).all(axis=1)].index
    df = df[~df[area_type].isin(low_neuron_areas)]
    print(f"Areas removed (neuron count < {thres}): {low_neuron_areas.tolist()}")

    # Filter by mouse count
    if mouse_thres > 0:
        mouse_counts = df.groupby(area_type)['mouse_id'].nunique()
        low_mouse_areas = mouse_counts[mouse_counts < mouse_thres].index
        df = df[~df[area_type].isin(low_mouse_areas)]
        print(f"Areas removed (mouse count < {mouse_thres}): {low_mouse_areas.tolist()}")
        
    return df
