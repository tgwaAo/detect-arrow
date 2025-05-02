#!/bin/env python3

from detectarrow.processing.preparation import Preparation


if __name__ == '__main__':
    preparator = Preparation()
    preparator.extract_raw_pos_imgs_from_videos()
    preparator.extract_pre_aug_imgs_from_big_imgs()
    preparator.aug_imgs_and_build_pos_dataset()

    preparator.extract_neg_candidates()
    preparator.move_part_of_unused_to_neg_dataset()
