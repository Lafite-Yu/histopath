import multiprocessing
from typing import Type
from matplotlib.pyplot import sca
import openslide
import numpy as np
import math
import PIL
from PIL import Image
Image.MAX_IMAGE_PIXELS = 20000000000
import os
from os.path import join as pjoin

from data_config import *
import utils


class Slide(object):
    def __init__(self, filepath):
        self.filepath = filepath
        self.filename = os.path.splitext(os.path.basename(filepath))[0]

        self.slide = utils.open_slide(filepath)
        associated_images = dict(self.slide.associated_images)
        self.thumbnail = associated_images['thumbnail'].convert('RGB')
        self.associated_label = associated_images['label'].convert('RGB')
        self.associated_macro = associated_images['macro'].convert('RGB')

        self.dimensions = self.slide.dimensions
        self.level_count = self.slide.level_count
        self.level_dimensions = self.slide.level_dimensions
        self.level_downsamples = [round(value) for value in self.slide.level_downsamples]
        
    def __str__(self):
        level_info = ['Downsample %sX, %s' % (self.level_downsamples[idx], self.level_dimensions[idx])  for idx in range(self.level_count)]
        info_str = f'[{self.filepath}]: {self.dimensions}\n'
        info_str += '\tLevels: %s\n' % ('\n\t\t'.join(level_info))
        info_str += f'\tAssociated thumbnail image: {self.thumbnail}\n'
        info_str += f'\tAssociated label image: {self.associated_label}\n'
        info_str += f'\tAssociated macro image: {self.associated_macro}'
        return info_str

    def pil_image(self, scale_factor):
        if scale_factor in self.level_downsamples:
            level = self.level_downsamples.index(scale_factor)
            image = self.slide.read_region(location=(0,0),
                                            level=level, 
                                            size=self.level_dimensions[level]).convert('RGB')
        else:
            level = self.slide.get_best_level_for_downsample()
            image = self.slide.read_region(location=(0,0),
                                            level=level, 
                                            size=self.level_dimensions[level]).convert('RGB')
            new_size = (round(self.level_dimensions[level][idx]/scale_factor) for idx in range(2))
            image = image.resize(new_size, Image.BILINEAR)
        return image

    def numpy_image(self, scale_factor):
        pil_image = self.pil_image(scale_factor)
        return np.asarray(pil_image)

    def save_converted_image(self, scale_factor, force_reconvert_when_exists=False):
        print(self.__str__())
        img_store_dir = pjoin(ARGS_converted_dir, os.path.splitext(self.filepath)[0])
        utils.exists_or_makedirs(img_store_dir)
        if not os.path.exists(pjoin(img_store_dir, 'macro.jpg')) or force_reconvert_when_exists :
            self.associated_macro.save(pjoin(img_store_dir, 'macro.jpg'))
        if not os.path.exists(pjoin(img_store_dir, 'label.jpg')) or force_reconvert_when_exists:
            self.associated_label.save(pjoin(img_store_dir, 'label.jpg'))
        if not os.path.exists(pjoin(img_store_dir, 'thumbnail.jpg')) or force_reconvert_when_exists:
            self.thumbnail.save(pjoin(img_store_dir, 'thumbnail.jpg'))
        if not os.path.exists(pjoin(img_store_dir, f'{self.filename}.{ARGS_convert_image_format}')) or force_reconvert_when_exists:
            try:
                self.pil_image(scale_factor).save(pjoin(img_store_dir, f'{self.filename}.{ARGS_convert_image_format}'))
            except openslide.lowlevel.OpenSlideError as e:
                print(repr(e))
        print(f'Slide image {self.filepath} saved.\n')
        
    def get_fullsize_tile(self, location, size):
        if isinstance(size, int):
            size = (size, size)
        elif not (isinstance(size, tuple) or isinstance(size, list)):
            raise TypeError('size should be list, tuple or int')
        if size[0]*size[1] >= 1024 * 1024:
            Warning('Too large tile required.')
        tile = self.slide.read_region(location=location, level=0, size=size).convert('RGB')
        return tile

    def get_fullsize_tile_by_bbox(self, upper_left, lower_rigth):
        return self.get_fullsize_tile(location=upper_left,
                                        size=(lower_rigth[0]-upper_left[0], lower_rigth[1]-upper_left[1]))

    def get_fullsize_region_by_tile(self, raw_location, raw_size, tile_size=1024):
        if isinstance(raw_size, int):
            raw_size = (raw_size, raw_size)
        elif not (isinstance(raw_size, tuple) or isinstance(raw_size, list)):
            raise TypeError('size should be list, tuple or int')

        width_tile_count, height_tile_count = math.ceil(raw_size[0]/tile_size), math.ceil(raw_size[1]/tile_size)
        last_tile_width = raw_size[0]%tile_size if raw_size[0]%tile_size!=0 else tile_size
        last_tile_height = raw_size[1]%tile_size if raw_size[1]%tile_size!=0 else tile_size
        image_array = np.zeros((raw_size[0], raw_size[1], 3), dtype=np.uint8)
            
        print(f'width_count{width_tile_count} height_count{height_tile_count}')

        anchor_point = [0,0]
        raw_img_pos = list(raw_location)
        for row in range(0,height_tile_count):
            for column in range(0,width_tile_count):
                tile_width = tile_size if column != width_tile_count-1 else last_tile_width
                tile_height = tile_size if row != height_tile_count-1  else last_tile_height
                image_array[anchor_point[0]:anchor_point[0]+tile_width, anchor_point[1]:anchor_point[1]+tile_height, :] = np.asarray(self.get_fullsize_tile(raw_img_pos, (tile_height, tile_width)))
                anchor_point[0] += tile_width
                raw_img_pos[0] += tile_width
            anchor_point[1] += tile_height
            anchor_point[0] = 0
            raw_img_pos[1] += tile_height
            raw_img_pos[0] = raw_location[0]
        return image_array

#TODO
    def get_scaled_region_by_tile(self, raw_location, raw_size, scale, tile_size=1024):
        if isinstance(raw_size, int):
            raw_size = (raw_size, raw_size)
        elif not (isinstance(raw_size, tuple) or isinstance(raw_size, list)):
            raise TypeError('size should be int, tuple or list')

        if scale in self.level_dimensions:
            raw_location = (round(raw_location[0]/scale), round(raw_location[1]/scale))
            level = self.level_dimensions.index(scale)
            raw_size = self.level_dimensions[level]
        else:
            level = 0 
        last_tile_width = raw_size[0]%tile_size if raw_size[0]%tile_size!=0 else tile_size
        last_tile_height = raw_size[1]%tile_size if raw_size[1]%tile_size!=0 else tile_size
        image_array = None
        width_tile_count, height_tile_count = math.ceil(raw_size[0]/tile_size), math.ceil(raw_size[1]/tile_size)
        raw_image_pos = list(raw_location)
        for row in range(0, height_tile_count):
            for column in range(0, width_tile_count):
                tile_width = tile_size if column!=width_tile_count-1 else last_tile_width
                tile_height = tile_size if row!=height_tile_count-1 else last_tile_height
                tile_image = self.get_fullsize_tile(raw_image_pos, (tile_width, tile_height))
                scaled_tile_width, scaled_tile_height = round(tile_width/scale), round(tile_height/scale)
                scaled_tile_image = tile_image.resize((scaled_tile_width, scaled_tile_height), Image.BILINEAR)
                scaled_tile_array = np.asarray(scaled_tile_image)
                #TODO
                raise NotImplementedError
                raw_image_pos[0] += tile_width
            raw_image_pos[0] = raw_location[0]
            raw_image_pos[1] += tile_height

    def get_fullsize_image_by_tile(self):
        return self.get_fullsize_region_by_tile(raw_location=(0,0), 
                                                raw_size=self.dimensions)

    def get_scaled_image_by_tile(self, scale):
        return self.get_scaled_region_by_tile(raw_location=(0,0),
                                            raw_size=self.level_dimensions,
                                            scale=scale)
