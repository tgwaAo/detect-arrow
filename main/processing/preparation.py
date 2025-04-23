from pathlib import PurePath
import pathlib as pl
import random
from shutil import copy2
from glob import glob
import itertools

import numpy as np
import cv2
from keras import Sequential
from keras import layers
# noinspection PyUnresolvedReferences
from keras.models import load_model
# noinspection PyUnresolvedReferences
from keras.preprocessing import image_dataset_from_directory

import numpy.typing as npt
from typing import Optional as Opt

from main.conf.paths import RAW_VIDS_PATH
from main.conf.paths import RAW_POS_IMGS_PATH
from main.conf.paths import ORIGINAL_POS_PATH
from main.conf.paths import ORIGINAL_POS_SUB_PATH
from main.conf.paths import ORIGINAL_NEG_PATH
from main.conf.paths import ORIGINAL_NEG_SUB_PATH
from main.conf.paths import DATASET_PATH
from main.conf.paths import ARROWS_PATH
from main.conf.paths import ANYTHING_PATH
from main.conf.paths import RAW_NEG_IMGS_PATH
from main.conf.paths import UNUSED_NEG_PATH
from main.conf.imgs import Colors
from main.conf.imgs import TARGET_SIZE
from main.conf.imgs import AREA_BORDER
from main.conf.imgs import BLUR_KERNEL
from main.conf.imgs import ARROW_CONTOUR_POINTS
from main.conf.ai import FRAMES_TILL_IMAGE_EXTRACTION
from main.conf.ai import AUG_RANDOM_ROTATION
from main.conf.ai import AUG_RANDOM_TRANSLATION_H
from main.conf.ai import AUG_RANDOM_TRANSLATION_W
from main.conf.ai import AUG_FILL_VALUE
from main.conf.ai import ROUGHLY_NBR_OF_FILES_PER_CLASS
from main.conf.ascii import Keys
from .utils import get_current_time_string
from .utils import save_img
from .utils import extract_cnts
from .utils import filter_cnts
from .utils import sort_cnts
from .utils import filter_and_extract_img_from_cnt
from .utils import extract_img_from_cnt
from .utils import choose_costum_path
from .utils import handle_options
from .utils import create_sub_path_with_nbr
from .model_handler import ModelHandler


class Preparation:
    def __init__(self):
        self.model = None
        self.videos_path = None
        self.ext_imgs_path = None
        self.org_pos_path = None
        self.org_pos_sub_path = None
        self.org_neg_path = None
        self.org_neg_sub_path = None
        self.dataset_path = None
        self.training_arrows_path = None
        self.training_anything_path = None
        self.path_idx = None

    def choose_costum_paths(self, single_choice=True):
        result = choose_costum_path(RAW_VIDS_PATH, only_existing=True)
        if result is None:
            return False
        self.videos_path, nbr = result

        if single_choice:
            self.path_idx = nbr
            self.ext_imgs_path = create_sub_path_with_nbr(RAW_POS_IMGS_PATH, nbr)
            self.org_pos_path = create_sub_path_with_nbr(ORIGINAL_POS_PATH, nbr)
            path = pl.PurePath(self.org_pos_path)
            self.org_pos_sub_path = str(path / path.name)
            self.org_neg_path = create_sub_path_with_nbr(ORIGINAL_NEG_PATH, nbr)
            path = pl.PurePath(self.org_neg_path)
            self.org_neg_sub_path = str(path / path.name)
            self.dataset_path = DATASET_PATH  # choosing another path results in a more difficult training setup

        else:
            result = choose_costum_path(RAW_POS_IMGS_PATH)
            if result is None:
                return False
            self.ext_imgs_path, _ = result

            result = choose_costum_path(ORIGINAL_POS_PATH)
            if result is None:
                return False
            self.org_pos_path, _ = result
            path = pl.PurePath(self.org_pos_path)
            self.org_pos_sub_path = path / path.name

            result = choose_costum_path(ORIGINAL_NEG_PATH)
            if result is None:
                return False
            self.org_neg_path, _ = result
            path = pl.PurePath(self.org_neg_path)
            self.org_neg_sub_path = path / path.name

            result = choose_costum_path(DATASET_PATH)
            if result is None:
                return False
            self.dataset_path, _ = result

        self.training_arrows_path = str(pl.PurePath(self.dataset_path) / pl.PurePath(ARROWS_PATH).name)
        self.training_anything_path = str(pl.PurePath(self.dataset_path) / pl.PurePath(ANYTHING_PATH).name)
        return True

    def check_contours_manually(
        self,
        cnts: npt.ArrayLike,
        gray_img: npt.ArrayLike,
        color: Colors,
        min_area_cnt: Opt[int] = None,
        org_neg_sub_path: Opt[str] = False
    ) -> npt.NDArray | None:
        pos_cnt = None
        neg_cnts = []
        for num, cnt in enumerate(cnts):
            min_rect = cv2.minAreaRect(cnt)
            area = min_rect[1][0] * min_rect[1][1]
            if min_area_cnt and area < min_area_cnt:
                continue
            col_frame = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(col_frame, cnts, num, color, thickness=10)
            cv2.imshow('Frame', col_frame)
            key = cv2.waitKey(0) & 0xFF
            if key == Keys.Y:
                if not org_neg_sub_path:
                    return cnt, None

            elif key == Keys.Q or key == Keys.ESC:
                return None
            elif key == Keys.R:
                return self.check_contours_manually(cnts, gray_img, color, min_area_cnt, org_neg_sub_path)
            elif org_neg_sub_path and key == Keys.N:
                neg_cnts.append(cnt)

        return pos_cnt, neg_cnts

    def extract_raw_pos_imgs_from_video(
            self,
            video_fname: str,
            nth_frame: int = FRAMES_TILL_IMAGE_EXTRACTION,
            ext_imgs_path: Opt[str] = None
    ) -> None:
        self.ext_imgs_path = handle_options(ext_imgs_path, self.ext_imgs_path, RAW_POS_IMGS_PATH)
        pl.Path(self.ext_imgs_path).mkdir(exist_ok=True)

        cap = cv2.VideoCapture(video_fname)
        if not cap.isOpened():
            cap.release()
            raise IOError(f'could not open {videos_fname}')

        counter = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if ret is False:
                    break

                if counter % nth_frame == 0:
                    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    save_img(self.ext_imgs_path, gray_img)

                counter += 1

        except Exception as e:
            print(e)
        cap.release()

    def extract_raw_pos_imgs_from_videos(
            self,
            videos_path: Opt[str] = None,
            nth_frame: int = FRAMES_TILL_IMAGE_EXTRACTION,
            ext_imgs_path: Opt[str] = None
    ) -> None:
        self.videos_path = handle_options(videos_path, self.videos_path, RAW_VIDS_PATH)
        self.ext_imgs_path = handle_options(ext_imgs_path, self.ext_imgs_path, RAW_POS_IMGS_PATH)

        for fname in glob(str(PurePath(self.videos_path, '*'))):
            self.extract_raw_pos_imgs_from_video(fname, nth_frame)

    def load_model_for_classification(self, model_path: str = None, ignore_when_model_exists: bool = True):
        if self.model is not None and ignore_when_model_exists:
            return True

        m_handler = ModelHandler()
        if model_path is None:
            ret = m_handler.load_model()
        else:
            ret = m_handler.load_model(model_path)

        if ret:
            self.model = m_handler.model
            m_handler.model = None
        else:
            return False

        self.model.trainable = False
        return True

    def extract_pos_imgs_from_imgs(
        self,
        use_model: bool = False,
        min_area_cnt: int = AREA_BORDER - 10,
        expected_pts: int = ARROW_CONTOUR_POINTS,
        ext_imgs_path: Opt[str] = None,
        org_pos_sub_path: Opt[str] = None,
        org_neg_sub_path: Opt[str] = None
    ):
        self.ext_imgs_path = handle_options(ext_imgs_path, self.ext_imgs_path, RAW_POS_IMGS_PATH)
        self.org_pos_sub_path = handle_options(org_pos_sub_path, self.org_pos_sub_path, ORIGINAL_POS_SUB_PATH)
        pl.Path(self.org_pos_sub_path).mkdir(parents=True, exist_ok=True)
        if org_neg_sub_path:
            pl.Path(org_neg_sub_path).mkdir(parents=True, exist_ok=True)

        for img_filename in pl.Path(self.ext_imgs_path).glob('*.jpg'):
            gray_img = cv2.imread(str(img_filename), cv2.IMREAD_GRAYSCALE)
            if gray_img is None:
                print(f'WARNING: {img_filename} could not be loaded')
                continue

            blurred = cv2.blur(gray_img, BLUR_KERNEL)
            cnts = extract_cnts(blurred)

            if use_model and self.model is not None:
                filtered_list, filtered_cnts, hull_rot_pts = filter_cnts(cnts, gray_img, expected_pts)
                if not len(filtered_list):
                    print(f'no candidate for prediction found in {img_filename}')
                    continue

                prediction = self.model.predict(filtered_list).flatten()
                pos_cnts, neg_cnts, _, _, _ = sort_cnts(prediction, filtered_cnts, hull_rot_pts)
            else:
                pos_cnts = []
                neg_cnts = cnts

            shown_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(shown_img, pos_cnts, -1, Colors.BLUE, 2)
            cv2.drawContours(shown_img, neg_cnts, -1, Colors.RED, 2)

            cv2.imshow('Frame', shown_img)
            key = cv2.waitKey(0) & 0xFF

            if key == Keys.N:
                pos_cnt, neg_cnts = self.check_contours_manually(
                    cnts,
                    gray_img,
                    Colors.PURPLE,
                    min_area_cnt,
                    org_neg_sub_path
                )
                if pos_cnt is not None:
                    small_img = extract_img_from_cnt(gray_img, pos_cnt)
                    save_img(self.org_pos_sub_path, small_img)

                if neg_cnts is not None:
                    for neg_cnt in neg_cnts:
                        small_img = extract_img_from_cnt(gray_img, neg_cnt)
                        save_img(org_neg_sub_path, small_img)

            elif key == Keys.Y:
                if len(pos_cnts) == 1:
                    small_img = extract_img_from_cnt(gray_img, pos_cnts[0])
                    save_img(self.org_pos_sub_path, small_img)

                else:
                    print('too many values -> no save')

            elif key == Keys.Q or key == 27:
                break

        cv2.destroyAllWindows()

    def build_aug_model(
        self,
        rnd_rot: float = AUG_RANDOM_ROTATION,
        w_factor: float = AUG_RANDOM_TRANSLATION_H,
        h_factor: float = AUG_RANDOM_TRANSLATION_W,
        fill_value: int = AUG_FILL_VALUE
    ) -> Sequential:
        return Sequential([
            layers.RandomRotation(rnd_rot),
            layers.RandomTranslation(h_factor, w_factor, fill_value=fill_value),
        ])

    def aug_imgs_and_build_pos_dataset(
        self,
        org_pos_path: Opt[str] = None,
        training_arrows_path: str = None,
        roughly_created_size: int = ROUGHLY_NBR_OF_FILES_PER_CLASS,
        batch_size: int = 1_000,
    ):
        self.org_pos_path = handle_options(org_pos_path, self.org_pos_path, ORIGINAL_POS_PATH)
        self.training_arrows_path = handle_options(training_arrows_path, self.training_arrows_path, ARROWS_PATH)
        pl.Path(self.training_arrows_path).mkdir(parents=True, exist_ok=True)
        return self.aug_imgs_and_build_dataset(
            self.org_pos_path,
            self.training_arrows_path,
            roughly_created_size,
            batch_size
        )

    def aug_imgs_and_build_neg_dataset(
            self,
            org_neg_path: Opt[str] = None,
            training_anything_path: str = None,
            roughly_created_size: int = ROUGHLY_NBR_OF_FILES_PER_CLASS,
            batch_size: int = 1_000
    ):
        self.org_neg_path = handle_options(org_neg_path, self.org_neg_path, ORIGINAL_NEG_PATH)
        self.training_anything_path = handle_options(training_anything_path, self.training_anything_path, ANYTHING_PATH)
        pl.Path(self.training_anything_path).mkdir(parents=True, exist_ok=True)
        return self.aug_imgs_and_build_dataset(
            self.org_neg_path,
            self.training_anything_path,
            roughly_created_size,
            batch_size
        )

    def aug_imgs_and_build_dataset(
        self,
        source_path: str,
        target_path: str,
        roughly_created_size,
        batch_size
    ):
        sub_paths = list(pl.Path(source_path).glob('*'))
        if len(sub_paths) != 1:
            print(f'only one sub path should exist, found {sub_paths} in {source_path}')
            return False

        path = pl.Path(sub_paths[0])
        nbr_files = len(list(path.iterdir()))
        if not nbr_files:
            print(f'warn: no files found in {path}')
            return False

        aug_net = self.build_aug_model()
        org_ds = image_dataset_from_directory(
            source_path,
            image_size=(TARGET_SIZE[1], TARGET_SIZE[0]),
            batch_size=batch_size
        )

        digits_in_fname = int(np.floor(np.log10(roughly_created_size)))
        nbr_augs_same_img = roughly_created_size / nbr_files
        if nbr_augs_same_img.is_integer():
            digits_in_fname += 1 # for equal dividable numbers that reach 10_000
        nbr_augs_same_img = int(np.floor(nbr_augs_same_img))

        num = 0
        for images_per_iter, label in org_ds:
            for _ in range(nbr_augs_same_img):
                aug_imgs = aug_net(images_per_iter).numpy()
                for img in aug_imgs:
                    filename = str(PurePath(
                        target_path,
                        f'img_{str(num).rjust(digits_in_fname, '0')}_{get_current_time_string()}.jpg'
                    ))
                    retval = cv2.imwrite(filename, img)
                    if not retval:
                        print(f'retval {retval} saving augmented image: {filename}')
                        return False
                    num += 1

        print(f'number of created images: {num}')
        return True

    def extract_neg_candidates(self):
        path = pl.Path(RAW_NEG_IMGS_PATH)
        path.mkdir(exist_ok=True)
        pl.Path(UNUSED_NEG_PATH).mkdir(exist_ok=True)

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
                    save_img(UNUSED_NEG_PATH, small_img)

            if num % 10 == 0:
                print(f'at least {num} raw images finished', end='\r')

    def move_part_of_unused_to_neg_dataset(self, nbr_imgs=ROUGHLY_NBR_OF_FILES_PER_CLASS):
        pl.Path(ANYTHING_PATH).mkdir(parents=True, exist_ok=True)
        used_neg_filenames = itertools.islice(pl.Path(UNUSED_NEG_PATH).glob('*.jpg'), nbr_imgs)
        for file in used_neg_filenames:
            copy2(file, PurePath(ANYTHING_PATH, file.name))
