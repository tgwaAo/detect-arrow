#!/bin/env python3

import pathlib as pl

import matplotlib.pyplot as plt
from keras.preprocessing import image_dataset_from_directory
from keras.layers import concatenate

from detectarrow.conf.paths import MODELS_PATH
from detectarrow.conf.paths import MODEL_BNAME
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

    # pos_roughly_created_size = get_nbr_of_imgs_for_aug(preparator.org_pos_sub_path, 'positive')
    # neg_roughly_created_size = get_nbr_of_imgs_for_aug(preparator.org_neg_sub_path, 'negative')
    # preparator.aug_imgs_and_build_pos_dataset(roughly_created_size=pos_roughly_created_size)
    # preparator.aug_imgs_and_build_neg_dataset(roughly_created_size=neg_roughly_created_size)
    #
    # default_model_bname = pl.PurePath(MODEL_BNAME)
    # saved_bname = pl.PurePath(f'{default_model_bname.stem}_{preparator.path_idx}{default_model_bname.suffix}')
    #
    # model_handler = ModelHandler()
    # model_handler.load_datasets()
    # model_handler.load_model()
    # model_handler.train_model(epochs=5)
    # model_handler.save_model(str(saved_bname))
    # model_handler.show_training_progress()
    # model_handler.prepare_validation()
    # model_handler.classification_report()
    # model_handler.saliency(import_hack=True)

