#!/bin/env python3

from pathlib import PurePath

videos_basepath = '../../raw-positive-videos/'
images_basepath = '../../raw-positive-images/'
neg_imgs_basepath = '../../raw-negative-images/'
unused_neg_basepath = '../../unused-negatives/'
original_pos_basepath = '../original-positives/'
dataset_basepath = '../dataset/'
arrows_basepath = str(PurePath(dataset_basepath, 'arrows/'))
model_basepath = '../model/'
examples_basepath = '../example-images/'
calibration_images_basepath = '../../calibration-images/'
camera_config_basepath = '../../cam-config/'


