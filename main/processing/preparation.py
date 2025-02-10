#!/bin/env python3

import glob
from pathlib import PurePath
import pathlib as pl
import random
from shutil import copy2

import numpy as np
import cv2
from keras import Sequential
from keras import layers
# noinspection PyUnresolvedReferences
from keras.models import load_model
# noinspection PyUnresolvedReferences
from keras.preprocessing import image_dataset_from_directory

import numpy.typing as npt

from main.conf.path_consts import images_basepath
from main.conf.path_consts import original_pos_basepath
from main.conf.path_consts import arrows_basepath
from main.conf.path_consts import neg_imgs_basepath
from main.conf.path_consts import unused_neg_basepath
from main.conf.img_consts import COLORS
from main.conf.img_consts import TARGET_SIZE
from main.conf.img_consts import BLUR_KERNEL
from utils import get_current_time_string
from utils import extract_cnts
from utils import filter_cnts
from utils import sort_cnts
from utils import filter_and_extract_img_from_cnt


class Preparation:
    def __init__(self):
        self.model = None

    def check_contours_manually(
        self,
        cnts: npt.ArrayLike,
        gray_img: npt.ArrayLike,
        color: COLORS
    ) -> npt.NDArray | None:
        for num, contour in enumerate(cnts):
            col_frame = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(col_frame, cnts, num, color, thickness=10)
            cv2.imshow('Frame', col_frame)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('y'):
                return contour
            elif key == ord('q') or key == 27:
                return None
            elif key == ord('r'):
                return self.check_contours_manually(cnts, gray_img, color)

        return None

    def extract_imgs_from_video(self, video_path: str, nth_frame: int) -> None:
        from datetime import datetime
        from datetime import timezone

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            cap.release()
            raise SystemExit(1)

        counter = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if ret is False:
                    break

                if counter % nth_frame == 0:
                    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    timestring = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S-%f')
                    retval = cv2.imwrite(str(PurePath(images_basepath, f'img_{timestring}.jpg')), gray_img)
                    if not retval:
                        raise ValueError(f'retval:{retval}')

                counter += 1

        except Exception as e:
            print(e)

        cap.release()

    def load_model_for_classification(self, model_path: str, ignore_when_model_exists: bool = True):
        if self.model is not None and ignore_when_model_exists:
            return False

        try:
            self.model = load_model(model_path)
        except Exception:
            return False

        self.model.trainable = False
        return True

    def extract_pos_imgs_from_imgs(self, use_model: bool = False):
        for img_filename in glob.glob(str(PurePath(images_basepath, '*.jpg'))):
            gray_img = cv2.imread(img_filename, cv2.IMREAD_GRAYSCALE)
            if gray_img is None:
                print(f'WARNING: {img_filename} could not be loaded')
                continue

            blurred = cv2.blur(gray_img, BLUR_KERNEL)
            cnts = extract_cnts(blurred)
            filtered_list, cnts, hull_rot_pts = filter_cnts(cnts, gray_img)
            if not len(filtered_list):
                print(f'no candidate for prediction found in {img_filename}')
                continue

            if use_model and self.model is not None:
                prediction = self.model.predict(filtered_list).flatten()
                pos_cnts, neg_cnts = sort_cnts(prediction, cnts)
            else:
                prediction = [False] * len(filtered_list)
                pos_cnts = []
                neg_cnts = cnts

            shown_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(shown_img, pos_cnts, -1, COLORS.BLUE, 2)
            cv2.drawContours(shown_img, neg_cnts, -1, COLORS.RED, 2)

            cv2.imshow('Frame', shown_img)
            key = cv2.waitKey(0) & 0xFF

            if key == ord('n'):
                contour = self.check_contours_manually(cnts, gray_img, COLORS.PURPLE)
                if contour is not None:
                    small_img = filter_and_extract_img_from_cnt(gray_img, contour, area_filter=False, w_h_filter=False)
                    timestring = get_current_time_string()
                    retval = cv2.imwrite(f'../original-positives/original-positives/img_{timestring}.jpg',
                                         small_img)
                    if not retval:
                        raise ValueError(f'retval:{retval}')

            elif key == ord('y'):
                if len(pos_cnts) == 1:
                    small_img = filter_and_extract_img_from_cnt(
                        gray_img, pos_cnts[0],
                        area_filter=False,
                        w_h_filter=False
                    )
                    timestring = get_current_time_string()
                    retval = cv2.imwrite(f'../original-positives/original-positives/img_{timestring}.jpg',
                                         small_img)
                    if not retval:
                        raise ValueError(f'retval:{retval}')

                else:
                    print('too many values -> no save')

            elif key == ord('q') or key == 27:
                break

        cv2.destroyAllWindows()

    def build_aug_model(self, rnd_rot: float, w_factor: float, h_factor: float, fill_value: int) -> Sequential:
        return Sequential([
            layers.RandomRotation(rnd_rot),
            layers.RandomTranslation(h_factor, w_factor, fill_value=fill_value),
        ])

    def aug_imgs_and_build_pos_dataset(self, aug_nbr: int = 40, batch_size: int = 1_000):
        aug_net = self.build_aug_model()

        org_ds = image_dataset_from_directory(
            original_pos_basepath,
            image_size=(TARGET_SIZE[1], TARGET_SIZE[0]),
            batch_size=batch_size
        )
        nbr_batches = org_ds.cardinality()
        max_nbr_elements = nbr_batches * batch_size
        est_pow_of_ten = np.floor(np.log10(max_nbr_elements)) + 1
        num = 0
        for all_images, label in org_ds:
            for _ in range(aug_nbr):
                aug_imgs = aug_net(all_images).numpy()
                for img in aug_imgs:
                    filename = str(PurePath(arrows_basepath, f'img_{str(num).rjust(est_pow_of_ten, '0')}.jpg'))
                    retval = cv2.imwrite(filename, img)
                    if not retval:
                        raise ValueError(f'retval saving augmented image:{retval}')
                    num += 1

            break  # avoid error message
        print(f'number of created images: {num}')

    def extract_neg_candidates(self):
        path = pl.Path(neg_imgs_basepath)
        for num, image_name in enumerate(path.iterdir()):
            big_neg_image = cv2.imread(str(image_name))
            if big_neg_image is None:
                print(f'could not load image {image_name}')
                break

            gray_img = cv2.cvtColor(big_neg_image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.blur(gray_img, BLUR_KERNEL)
            cnts = extract_cnts(blurred)

            for con in cnts:
                small_img = filter_and_extract_img_from_cnt(gray_img, con)
                if small_img is not None:
                    timestring = get_current_time_string()
                    retval = cv2.imwrite(str(PurePath(unused_neg_basepath, f'img_{timestring}.jpg')), small_img)
                    if not retval:
                        raise ValueError(f'retval saving negative image:{retval}')

            if num % 10 == 0:
                print(f'at least {num} raw images finished', end='\r')

    def move_part_of_unused_to_neg_dataset(self, nbr_imgs=10_000):
        used_neg_filenames = random.sample(glob.glob(str(PurePath(unused_neg_basepath, '*.jpg'))), k=nbr_imgs)
        for filename in used_neg_filenames:
            copy2(filename, f'../dataset/anything/{os.path.basename(filename)}')