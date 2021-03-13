import os
from os.path import join as pjoin
import xml
import datetime
import openslide

from data_config import *


def get_dataset_item_list(root=ARGS_raw_dir):
    def __dataset_walk(root, folder):
        try:
            folder_contents = os.listdir(pjoin(root, folder))
        except NotADirectoryError:
            return [folder]
        else:
            ret_items = []
            for path in folder_contents:
                ret_items.extend(__dataset_walk(root, pjoin(folder, path)))
        return ret_items
    
    folder_contents = os.listdir(root)
    ret_items = []
    for path in folder_contents:
        ret_items.extend(__dataset_walk(root, path))
    return ret_items


def get_raw_item_path_with_index(index):
    dataset_item_list = get_dataset_item_list(ARGS_raw_dir)
    return dataset_item_list[index]


def open_slide(filepath):
    full_path = pjoin(ARGS_raw_dir, filepath)
    return openslide.open_slide(full_path)


def open_annotation(filepath):
    filepath = os.path.splitext(filepath)[0]+'.xml'
    fullpath = pjoin(ARGS_annotation_dir, filepath)
    dom_tree = xml.dom.minidom.parse(fullpath)
    root_node = dom_tree.documentElement
    return root_node


def get_converted_dir_by_scale_factor(scale_factor):
    return pjoin(ARGS_dataset_dir, 'preprocessed', ARGS_dataset_chosen, f'{scale_factor}X')


def exists_or_makedirs(filepath):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
        return False
    else:
        return True


class Time:
  """
  Class for displaying elapsed time.
  """

  def __init__(self):
    self.start = datetime.datetime.now()

  def elapsed_display(self):
    time_elapsed = self.elapsed()
    print("Time elapsed: " + str(time_elapsed))

  def elapsed(self):
    self.end = datetime.datetime.now()
    time_elapsed = self.end - self.start
    return time_elapsed
