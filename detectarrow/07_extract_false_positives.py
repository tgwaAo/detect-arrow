#!/bin/env python3

import pathlib as pl
import cv2

from conf.paths import BIG_NEG_IMGS_PATH
from conf.paths import ORIGINAL_NEG_PATH
from conf.paths import MODEL_BNAME
from conf.imgs import BLUR_KERNEL
from conf.imgs import ARROW_CONTOUR_POINTS
from processing.model_handler import ModelHandler
from processing.utils import extract_cnts
from processing.utils import filter_cnts
from processing.utils import sort_cnts
from processing.utils import save_img
from processing.utils import choose_costum_path
from processing.utils import get_nbr_of_imgs_for_aug
from processing.utils import srtd_lst_candidates
from processing.utils import costum_sort
from processing.preparation import Preparation
if __name__ == '__main__':
    input_description = f'insert source path [{BIG_NEG_IMGS_PATH}]>>'
    source_path = input(input_description)
    if not source_path:
        source_path = pl.Path(BIG_NEG_IMGS_PATH)
    else:
        source_path = pl.Path(source_path)
    org_neg_path = pl.Path(ORIGINAL_NEG_PATH)
    tmp_path = srtd_lst_candidates(org_neg_path)[-1]
    working_idx = costum_sort(tmp_path)
    org_neg_path = pl.Path(org_neg_path.parent, f'{org_neg_path}-{working_idx + 1}')
    pre_aug_desc = f'choose output path [{org_neg_path}]>>'
    pre_aug_path = input(pre_aug_desc)
    if not pre_aug_path:
        pre_aug_path = org_neg_path
    else:
        pre_aug_path = pl.Path(pre_aug_path)

    pre_aug_sub_path = pre_aug_path / pre_aug_path.name
    pre_aug_sub_path.mkdir(parents=True, exist_ok=True)

    model_handler = ModelHandler()
    model_handler.load_model()
    for img_fname in source_path.glob('*.jpg'):
        gray_img = cv2.imread(str(img_fname), cv2.IMREAD_GRAYSCALE)
        blurred = cv2.blur(gray_img, BLUR_KERNEL)
        cnts = extract_cnts(blurred)

        filtered_list, filtered_cnts, hull_rot_pts = filter_cnts(cnts, gray_img, ARROW_CONTOUR_POINTS)
        if not len(filtered_list):
            print(f'no candidate for prediction found in {img_fname}')
            continue

        prediction = model_handler.model.predict(filtered_list).flatten()
        for working_idx, single_pred in enumerate(prediction):
            if single_pred >= 0.5:
                save_img(str(pre_aug_sub_path), filtered_list[working_idx])

    neg_roughly_created_size = get_nbr_of_imgs_for_aug(str(pre_aug_sub_path), 'negative')
    preparator = Preparation()
    preparator.aug_imgs_and_build_neg_dataset(
        org_neg_path=str(pre_aug_path),
        roughly_created_size=neg_roughly_created_size
    )

    default_model_bname = pl.PurePath(MODEL_BNAME)
    saved_bname = pl.PurePath(f'{default_model_bname.stem}_{working_idx}{default_model_bname.suffix}')
    model_handler = ModelHandler()
    model_handler.load_dataset()
    model_handler.load_model()
    model_handler.train_model(epochs=5)
    model_handler.save_model(str(saved_bname))
    model_handler.show_training_progress()
    model_handler.prepare_validation()
    model_handler.classification_report()
    model_handler.saliency(import_hack=True)
