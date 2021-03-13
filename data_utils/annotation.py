from math import isclose
import numpy as np
import cv2

import utils
from slide import Slide
from data_config import *

COLORS = [(178, 34, 34), (0, 128, 0)]
GRAY_SCALE_COLORS = [128,255]

class Annotation:
    def __init__(self, path):
        self.filepath = path
        self.slide = Slide(path)
        self.ASAP_Annotations = utils.open_annotation(path)
        self.AnnotationGroups = self._get_annoation_groups()
        self.Annotations = self._get_annotations()

    def _get_annoation_groups(self):
        groups_list = []
        for groups in self.ASAP_Annotations.getElementsByTagName('AnnotationGroups'):
            for group in groups.getElementsByTagName('Group'):
                name = group.getAttribute('Name')
                assert name in ['positive', 'Positive', '1', 'pos', 'negative', 'Negative', '0', 'neg']
                assert group.getAttribute('PartOfGroup') == 'None'
                groups_list.append(name)
        return groups_list
    

    def _get_annotations(self):
        annotations_list = []
        for annotations in self.ASAP_Annotations.getElementsByTagName('Annotations'):
            for annotation in annotations.getElementsByTagName('Annotation'):
                annotation_dict = {'name': annotation.getAttribute('Name'),
                                    'type': annotation.getAttribute('Type'),
                                    'label': annotation.getAttribute('PartOfGroup'),
                                    'coordinates': None,
                                    'bbox': None}
                assert annotation_dict['type'] in ['Rectangle', 'Spline', 'Polygon']
                if annotation_dict['label'] in ['Positive', 'positive', '1', 'pos']:
                    annotation_dict['label'] = 1
                elif annotation_dict['label'] in ['Negative', 'negative', '0', 'neg', 'None']:
                    annotation_dict['label'] = 0
                else:
                    raise ValueError(f'Annotation{annotation_dict["name"]} wrong PartOfGroup: {annotation_dict["label"]}')

                coordinate_list = []
                for coordinate in annotation.getElementsByTagName('Coordinate'):
                    coordinate_list.append([max(float(coordinate.getAttribute('X')), 0), 
                                            max(float(coordinate.getAttribute('Y')), 0)])
                coordinate_list = np.asarray(coordinate_list)
                annotation_dict['coordinates'] = coordinate_list

                bbox = ((np.min(coordinate_list[:, 0]), np.min(coordinate_list[:, 1])), 
                            (np.max(coordinate_list[:, 0]), np.max(coordinate_list[:, 1])))
                annotation_dict['bbox'] = bbox

                annotations_list.append(annotation_dict)
        return annotations_list

    def get_thumbnail_array(self):
        return self.get_annotated_array(-1)

    def get_annotated_array(self, scale_factor):
        if scale_factor == -1:
            image = self.slide.thumbnail
            scale_factor = self.slide.dimensions[0] / self.slide.thumbnail.size[0]
        else:
            image = self.slide.pil_image(scale_factor=scale_factor)
            scale_factor = self.slide.dimensions[0] / image.size[0]
        image_array = np.asarray(image)
        for annotation in self.Annotations:
            pt1 = tuple([round(value/scale_factor) for value in annotation['bbox'][0]])
            pt2 = tuple([round(value/scale_factor) for value in annotation['bbox'][1]])
            cv2.rectangle(image_array, pt1=pt1, pt2=pt2, color=COLORS[annotation['label']], thickness=round(100/scale_factor))
            if annotation['type'] != 'Rectangle':
                pts = np.array([annotation['coordinates']/scale_factor], dtype=np.int32)
                cv2.polylines(image_array,  pts=pts, isClosed=True, color=COLORS[annotation['label']], thickness=round(100/scale_factor))
        return image_array

    def get_mask_image(self, scale_factor, include_bbox=False):
        mask_array_dimensions = [round(value/scale_factor) for value in reversed(self.slide.dimensions)]
        mask_array = np.zeros(mask_array_dimensions, dtype=np.uint8)
        for annotation in self.Annotations:
            if annotation['type'] in ['Spline', 'Polygon']:
                pts = np.array([annotation['coordinates']/scale_factor], dtype=np.int32)
                cv2.fillPoly(mask_array, pts, color=GRAY_SCALE_COLORS[annotation['label']])
            elif annotation['type'] == 'Rectangle' and include_bbox:
                pt1 = tuple([round(value/scale_factor) for value in annotation['bbox'][0]])
                pt2 = tuple([round(value/scale_factor) for value in annotation['bbox'][1]])
                cv2.rectangle(mask_array, pt1=pt1, pt2=pt2, color=GRAY_SCALE_COLORS[annotation['label']], thickness=-1)
        return mask_array

    def get_mask_tile(self, position, size, include_bbox=False):
        if isinstance(size, int):
            size = (size, size)
        elif not (isinstance(size, tuple) or isinstance(size, list)):
            raise TypeError('size should be int, tuple or list')

        if position[0] > self.slide.dimensions[0] or position[1] > self.slide.dimensions[1]:
            raise OverflowError(f'Required position {position} is out of slide_dimensions {self.slide.dimensions}')

        mask_array = self.get_mask_image(1, include_bbox)
        mask_tile_array = mask_array[position[0]:position[0]+size[0], position[1]:position[1]+size[1]]
        return mask_tile_array
        
    def get_annotated_image_tile(self, position, size, level):
        if isinstance(size, int):
            size = (size, size)
        elif not (isinstance(size, tuple) or isinstance(size, list)):
            raise TypeError('size should be int, tuple or list')

        if position[0] > self.slide.level_dimensions[level][0] or position[1] > self.slide.level_dimensions[level][1]:
            raise OverflowError(f'Required position {position} is out of dimension of level {level}: {self.slide.level_dimensions[level]}')

        image_array = self.get_annotated_array(self.slide.level_downsamples[level])
        tile_array = image_array[position[0]:position[0]+size[0], position[1]:position[1]+size[1], :]
        return tile_array


