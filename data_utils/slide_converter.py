import multiprocessing
from os.path import getsize
import openslide
import numpy as np
import math
import PIL
from PIL import Image
Image.MAX_IMAGE_PIXELS = 20000000000
import random
from matplotlib import pyplot as plt

from data_config import *
from slide import Slide
import utils


class SlideConverter:
    def __init__(self, filepath_list, scale_factor):
        self.filepath_list = filepath_list
        self.scale_factor = scale_factor
        
    def _converter(self, slide_paths_list):
        for slide_path in slide_paths_list:
            print('Converting %s' % slide_path)
            slide = Slide(slide_path)
            slide.save_converted_image(self.scale_factor)

    def multithread_convert(self, num_thread=0):
        timer = utils.Time()

        # how many processes to use
        num_thread = multiprocessing.cpu_count() if num_thread==0 else num_thread
        pool = multiprocessing.Pool(num_thread)

        num_images = len(self.filepath_list)
        num_thread = min(num_thread, num_images)
        images_per_process = math.ceil(num_images / num_thread)

        print(f"Number of processes: {num_thread}")
        print(f"Number of training images: {num_images}")

        idx_list = list(range(num_images))
        random.shuffle(idx_list)

        # each task specifies a range of slides
        results = []
        for i in range(0, num_images, images_per_process):
            slides_list = [self.filepath_list[idx] for idx in sorted(idx_list[i:i+images_per_process])]
            results.append(pool.apply_async(self._converter, [slides_list]))
        
        for result in results:
            result.get()

        timer.elapsed_display()


def convert_all_slides(scale_factor, num_thread=0):
    converter = SlideConverter(utils.get_dataset_item_list(),
                                scale_factor)
    converter.multithread_convert(num_thread)
        

def get_stats(scale_factor):
    if scale_factor == 0:
        path = ARGS_raw_dir
    else:
        path = utils.get_converted_dir_by_scale_factor(scale_factor)
    
    item_list = [pjoin(path, file) for file in utils.get_dataset_item_list()]
    
    w_list = []
    h_list = []
    shape_list = []
    file_size_list = []
    
    utils.exists_or_makedirs(ARGS_stat_dir)

    
    with open(pjoin(ARGS_stat_dir, 
                '%s_file_info.log' % ('raw' if scale_factor==0 else f'{scale_factor}X')), 'w') as f:
        for item_path in item_list:
            if scale_factor == 0:
                slide = Slide(item_path)
                f.write(str(slide))
                w, h = slide.dimensions
                file_size_list.append(os.path.getsize(pjoin(ARGS_raw_dir, item_path)) / 1024**2)
            else:
                image_path = pjoin(utils.get_converted_dir_by_scale_factor(scale_factor),
                                    os.path.splitext(item_path)[0],
                                    os.path.basename(os.path.splitext(item_path)[0])+'.'+ARGS_convert_image_format)
                try:
                    image = Image.open(image_path)
                except FileNotFoundError as e:
                    print(e)
                    continue
                w, h = image.size
                file_size_list.append(os.path.getsize(image_path) / 1024**2)
            w_list.append(w)
            h_list.append(h)
            shape_list.append(w*h)

        w_list = np.array(w_list)
        h_list = np.array(h_list)
        shape_list = np.array(shape_list)
        file_size_list = np.array(file_size_list)

        info_str = f'Max image shape file:\t#{shape_list.argmax()} {item_list[int(shape_list.argmax())]}: {shape_list.max()} {w_list[int(shape_list.argmax())]}x{h_list[int(shape_list.argmax())]}\n'
        info_str += f'Min image shape file:\t#{shape_list.argmin()} {item_list[int(shape_list.argmin())]}: {shape_list.min()} {w_list[int(shape_list.argmin())]}x{h_list[int(shape_list.argmin())]}\n'
        info_str += f'Max size file:\t#{file_size_list.argmax()} {item_list[int(file_size_list.argmax())]}: {file_size_list.max()}MB {w_list[int(file_size_list.argmax())]}x{h_list[int(file_size_list.argmax())]}\n'
        info_str += f'Min size file:\t#{file_size_list.argmin()} {item_list[int(file_size_list.argmin())]}: {file_size_list.min()}MB {w_list[int(file_size_list.argmin())]}x{h_list[int(file_size_list.argmin())]}\n'
        info_str += f'Max width file:\t#{w_list.argmax()} {item_list[int(w_list.argmax())]}: {w_list.max()}x{h_list[int(w_list.argmax())]}\n'
        info_str += f'Min width file:\t#{w_list.argmin()} {item_list[int(w_list.argmin())]}: {w_list.min()}x{h_list[int(w_list.argmin())]}\n'
        info_str += f'Max height file:\t#{h_list.argmax()} {item_list[int(h_list.argmax())]}: {w_list[int(h_list.argmax())]}x{h_list.max()}\n'
        info_str += f'Min height file:\t#{h_list.argmin()} {item_list[int(h_list.argmin())]}: {w_list[int(h_list.argmin())]}x{h_list.min()}\n'
        print(info_str)
        f.write(info_str)

        colors = np.random.rand(len(w_list))
        plt.scatter(w_list, h_list, s=10, c=colors, alpha=0.7)
        plt.xlabel("width (pixels)")
        plt.ylabel("height (pixels)")
        plt.title("SVS Image Sizes")
        plt.set_cmap("prism")
        plt.tight_layout()
        plt.savefig(pjoin(ARGS_stat_dir, "%s-image-sizes.jpg" % ('raw' if scale_factor==0 else f'{scale_factor}X')))

        plt.clf()
        plt.hist(shape_list/1000000, bins=64)
        plt.xlabel("width x height (M of pixels)")
        plt.ylabel("# images")
        plt.title("Distribution of image shape in millions of pixels")
        plt.tight_layout()
        plt.savefig(pjoin(ARGS_stat_dir, "%s-distribution-of-image-shapes.jpg" % ('raw' if scale_factor==0 else f'{scale_factor}X')))

        plt.clf()
        plt.hist(file_size_list, bins=64)
        plt.xlabel("Image size(MB)")
        plt.ylabel("# images")
        plt.title("Distribution of image file sizes in MB")
        plt.tight_layout()
        plt.savefig(pjoin(ARGS_stat_dir, "%s-distribution-of-image-file-sizes.jpg" % ('raw' if scale_factor==0 else f'{scale_factor}X')))

if __name__ == '__main__':
    # convert_all_slides(ARGS_scale_factor, num_thread=16)
    get_stats(scale_factor=ARGS_scale_factor)

    # converter = SlideConverter(utils.get_dataset_item_list()[:5],
    #                             scale_factor=ARGS_scale_factor)
    # converter._converter(converter.filepath_list)