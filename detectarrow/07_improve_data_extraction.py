#!/bin/env python3

import pathlib as pl

import matplotlib.pyplot as plt
from keras.preprocessing import image_dataset_from_directory
from keras.layers import concatenate

from conf.paths import MODELS_PATH
from conf.paths import MODEL_BNAME
from conf.paths import ORIGINAL_NEG_SUB_PATH
from conf.paths import DATASET_PATH
from conf.imgs import COMPARED_SIZE
from processing.preparation import Preparation
from processing.model_handler import ModelHandler
from processing.utils import get_nbr_of_imgs_for_aug

if __name__ == '__main__':
    preparator = Preparation()
    preparator.choose_costum_paths()
    preparator.extract_raw_pos_imgs_from_videos()
    try:
        preparator.load_model_for_classification()
    except ValueError:
        print(f'could not load model {default_model_fname}')
        exit(2)
    except Exception as e:
        print(e)
        exit(3)

    preparator.extract_pre_aug_imgs_from_big_imgs(use_model=True)
    preparator.model = None

    pos_roughly_created_size = get_nbr_of_imgs_for_aug(preparator.org_pos_sub_path, 'positive')
    neg_roughly_created_size = get_nbr_of_imgs_for_aug(preparator.org_neg_sub_path, 'negative')
    preparator.aug_imgs_and_build_pos_dataset(roughly_created_size=pos_roughly_created_size)
    preparator.aug_imgs_and_build_neg_dataset(roughly_created_size=neg_roughly_created_size)

    default_model_bname = pl.PurePath(MODEL_BNAME)
    saved_bname = pl.PurePath(f'{default_model_bname.stem}_{preparator.path_idx}{default_model_bname.suffix}')

    model_handler = ModelHandler()
    model_handler.load_dataset()
    model_handler.load_model()
    model_handler.train_model(epochs=5)
    model_handler.save_model(str(saved_bname))
    model_handler.show_training_progress()
