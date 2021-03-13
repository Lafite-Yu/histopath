import os
from os.path import join as pjoin

ARGS_scale_factor = 16
ARGS_convert_image_format = 'png'

ARGS_dataset_dir = r'/home/LAB/yujinze/workspace/histopath/data'
ARGS_dataset_chosen = 'LN20210301_201slides'
ARGS_raw_dir = pjoin(ARGS_dataset_dir, 'raw', ARGS_dataset_chosen)
ARGS_converted_dir = pjoin(ARGS_dataset_dir, 'preprocessed', ARGS_dataset_chosen, f'{ARGS_scale_factor}X')
ARGS_stat_dir = pjoin(ARGS_dataset_dir, 'stats', ARGS_dataset_chosen)
ARGS_annotation_dir = pjoin(ARGS_dataset_dir, 'annotation', ARGS_dataset_chosen)
