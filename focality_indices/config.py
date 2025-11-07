import os

#File Paths
BASE_PATH = "TEMP_PATH_FOR_REPO"
DATA_PATH = os.path.join(BASE_PATH, "TEMP_PATH_FOR_REPO")
FIGURE_PATH = os.path.join(BASE_PATH, "TEMP_PATH_FOR_REPO")
MOUSE_INFO_PATH = os.path.join(BASE_PATH, "TEMP_PATH_FOR_REPO")

# Analysis Parameters
SINGLE_ANALYSES = [
    'auditory_active', 'auditory_passive_pre', 'auditory_passive_post',
    'baseline_choice', 'baseline_whisker_choice', 'choice', 'spontaneous_licks',
    'wh_vs_aud_pre_vs_post_learning', 'whisker_active', 'whisker_passive_pre',
    'whisker_passive_post', 'whisker_choice'
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
