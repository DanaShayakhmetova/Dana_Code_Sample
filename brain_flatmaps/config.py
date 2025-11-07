"""
Configuration file for the brain flatmap plotting section of the project.
"""
import os

# Project Colors
PROJECT_COLORS = {
    # single analyses and one colorbar
    'whisker': 'forestgreen',
    'auditory': 'mediumblue',
    'spontaneous': '#FF8C42',
    'choice': 'darkorchid',
    'baseline_choice': 'chocolate',
    'learning': 'mediumvioletred',

    # general reward groups two colorbars
    'rewarded': 'forestgreen',
    'non_rewarded': 'crimson',

    # analyses x reward groups two colorbars
    'whisker_rewarded': 'forestgreen',
    'whisker_nonrewarded': 'mediumseagreen',
    'auditory_rewarded': 'mediumblue',
    'auditory_nonrewarded': 'cornflowerblue',
    'learning_rewarded': 'darkmagenta',
    'learning_nonrewarded': 'indigo',
    'spontaneous_rewarded': 'teal',
    'spontaneous_nonrewarded': 'darkturquoise',
    'choice_rewarded': 'darkorchid',
    'choice_nonrewarded': 'mediumorchid',
    'baseline_choice_rewarded': 'chocolate',
    'baseline_choice_nonrewarded': 'sandybrown',
    'base': 'blues'
}

#Plotting Constants 
LIGHT_GRAY = '#D3D3D3'
DARK_GRAY = "#7E7D7D"
MISSING_DATA_DARK = -2.0  # Special value for regions with units but no significant ones

# File Paths 
BASE_PATH = "TEMP_FOR_THIS_EXAMPLE"
DATA_PATH = os.path.join("TEMP_FOR_THIS_EXAMPLE")
FIGURE_PATH = os.path.join("TEMP_FOR_THIS_EXAMPLE")
MOUSE_INFO_PATH = os.path.join("TEMP_FOR_THIS_EXAMPLE")


# Analysis Groups
SINGLE_ANALYSES = [
    'auditory_active', 'auditory_passive_pre', 'auditory_passive_post',
    'baseline_choice', 'baseline_whisker_choice', 'choice', 'spontaneous_licks',
    'wh_vs_aud_pre_vs_post_learning', 'whisker_active', 'whisker_passive_pre',
    'whisker_passive_post', 'whisker_choice', 'whisker_pre_vs_post_learning',
    'auditory_pre_vs_post_learning'
]

REWARD_GROUPS = ['R+', 'R-']

PAIR_ANALYSES = [
    ['whisker_active', 'auditory_active'],
    ['whisker_passive_pre', 'whisker_passive_post'],
    ['choice', 'whisker_choice'],
    ['baseline_whisker_choice', 'whisker_choice'],
    ['auditory_passive_pre', 'auditory_passive_post'],
    ['baseline_choice', 'choice']
]

DELTA_PAIRS = [
    ['whisker_passive_pre', 'whisker_passive_post'],
    ['auditory_passive_pre', 'auditory_passive_post']
]
