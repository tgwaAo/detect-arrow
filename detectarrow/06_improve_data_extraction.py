#!/bin/env python3

import pathlib as pl

import matplotlib.pyplot as plt
from keras.preprocessing import image_dataset_from_directory
from keras.layers import concatenate

from detectarrow.conf.paths import MODELS_PATH
from detectarrow.conf.paths import MODEL_BNAME
from detectarrow.conf.paths import ORIGINAL_NEG_PATH
from detectarrow.conf.paths import ORIGINAL_NEG_SUB_PATH
from detectarrow.conf.paths import DATASET_PATH
from detectarrow.conf.imgs import COMPARED_SIZE
from detectarrow.processing.preparation import Preparation
from detectarrow.processing.model_handler import ModelHandler


def get_nbr_of_imgs_for_aug(path: str, text: str):
    nbr_files = len(list(pl.Path(path).iterdir()))
    print(f'got {nbr_files} files in {path}')
    ans = input(f'roughly created size for {text} dataset [None] >>')
    if ans.isdigit():
        return int(ans)
    return None


if __name__ == '__main__':
    # default_model_fname = str(pl.PurePath(MODELS_PATH, MODEL_BNAME))
    # preparator = Preparation()
    # result = preparator.choose_costum_paths()
    # if not result:
    #     exit(1)

    # preparator.extract_raw_pos_imgs_from_videos()
    # preparator.load_model_for_classification(default_model_fname)
    # preparator.extract_pos_imgs_from_imgs(use_model=True)
    #
    # pos_roughly_created_size = get_nbr_of_imgs_for_aug(preparator.org_pos_sub_path, 'positive')
    # neg_roughly_created_size = get_nbr_of_imgs_for_aug(preparator.org_neg_sub_path, 'negative')
    # _ = preparator.aug_imgs_and_build_pos_dataset(roughly_created_size=pos_roughly_created_size)
    # _ = preparator.aug_imgs_and_build_neg_dataset(roughly_created_size=neg_roughly_created_size)
    #
    # saved_bname = pl.PurePath(MODEL_BNAME)
    # saved_bname = f'{saved_bname.stem}_{preparator.path_idx}{saved_bname.suffix}'
    model_handler = ModelHandler()
    model_handler.load_model()
    model_handler.load_datasets()
    # model_handler.train_model(epochs=5)
    # model_handler.save_model(saved_bname)
    model_handler.classification_report()
    model_handler.saliency(import_hack=True)

