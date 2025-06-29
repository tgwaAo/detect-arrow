from pathlib import PurePath
import pathlib as pl
import random
from shutil import move
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
from typing import Union

from conf.paths import RAW_VIDS_PATH
from conf.paths import RAW_IMGS_PATH
from conf.paths import ORIGINAL_POS_PATH
from conf.paths import ORIGINAL_POS_SUB_PATH
from conf.paths import ORIGINAL_NEG_PATH
from conf.paths import ORIGINAL_NEG_SUB_PATH
from conf.paths import DATASET_PATH
from conf.paths import ARROWS_PATH
from conf.paths import ANYTHING_PATH
from conf.paths import BIG_NEG_IMGS_PATH
from conf.paths import UNUSED_NEG_PATH
from conf.imgs import Colors
from conf.imgs import TARGET_SIZE
from conf.imgs import AREA_BORDER
from conf.imgs import BLUR_KERNEL
from conf.imgs import ARROW_CONTOUR_POINTS
from conf.ai import FRAMES_TILL_IMAGE_EXTRACTION
from conf.ai import AUG_RANDOM_ROTATION
from conf.ai import AUG_RANDOM_TRANSLATION_H
from conf.ai import AUG_RANDOM_TRANSLATION_W
from conf.ai import AUG_FILL_VALUE
from conf.ai import ROUGHLY_NBR_OF_FILES_PER_CLASS
from conf.ascii import Keys
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

type cnt_container = Union[
    list[npt.NDArray[int]],
    tuple[npt.NDArray[int]]
]


class Preparation:
    def __init__(self) -> None:
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

    def choose_costum_paths(self, single_choice: bool = True) -> None:
        result = choose_costum_path(RAW_VIDS_PATH, only_existing=True)
        self.videos_path, nbr = result
        if single_choice:
            self.path_idx = nbr
            self.ext_imgs_path = create_sub_path_with_nbr(RAW_IMGS_PATH, nbr)
            self.org_pos_path = create_sub_path_with_nbr(ORIGINAL_POS_PATH, nbr)
            path = pl.PurePath(self.org_pos_path)
            self.org_pos_sub_path = str(path / path.name)
            self.org_neg_path = create_sub_path_with_nbr(ORIGINAL_NEG_PATH, nbr)
            path = pl.PurePath(self.org_neg_path)
            self.org_neg_sub_path = str(path / path.name)
            self.dataset_path = DATASET_PATH  # choosing another path results in a more difficult training setup

        else:
            result = choose_costum_path(RAW_IMGS_PATH)
            self.ext_imgs_path, _ = result

            result = choose_costum_path(ORIGINAL_POS_PATH)
            self.org_pos_path, _ = result
            path = pl.PurePath(self.org_pos_path)
            self.org_pos_sub_path = path / path.name

            result = choose_costum_path(ORIGINAL_NEG_PATH)
            self.org_neg_path, _ = result
            path = pl.PurePath(self.org_neg_path)
            self.org_neg_sub_path = path / path.name

            result = choose_costum_path(DATASET_PATH)
            self.dataset_path, _ = result

        self.training_arrows_path = str(pl.PurePath(self.dataset_path) / pl.PurePath(ARROWS_PATH).name)
        self.training_anything_path = str(pl.PurePath(self.dataset_path) / pl.PurePath(ANYTHING_PATH).name)

    def check_contours_manually(
        self,
        cnts: cnt_container,
        gray_img: npt.ArrayLike,
        color: Colors,
        min_area_cnt: Opt[int] = None,
    ) -> Opt[tuple[Opt[npt.NDArray[int]], list[npt.NDArray[int]]]]:
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
                # noinspection PyTypeChecker
                return cnt, neg_cnts
            elif key == Keys.Q or key == Keys.ESC:
                return None
            elif key == Keys.R:
                return self.check_contours_manually(cnts, gray_img, color, min_area_cnt)
            elif key == Keys.N:
                neg_cnts.append(cnt)

        # noinspection PyTypeChecker
        return None, neg_cnts

    def extract_raw_pos_imgs_from_video(
            self,
            video_fname: str,
            nth_frame: int = FRAMES_TILL_IMAGE_EXTRACTION,
            ext_imgs_path: Opt[str] = None
    ) -> None:
        self.ext_imgs_path = handle_options(ext_imgs_path, self.ext_imgs_path, RAW_IMGS_PATH)
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
        self.ext_imgs_path = handle_options(ext_imgs_path, self.ext_imgs_path, RAW_IMGS_PATH)

        for fname in glob(str(PurePath(self.videos_path, '*'))):
            self.extract_raw_pos_imgs_from_video(fname, nth_frame)

    def load_model_for_classification(self, model_bname: str = None, ignore_when_model_exists: bool = True) -> None:
        if self.model is not None and ignore_when_model_exists:
            return
        model_handler = ModelHandler()
        model_handler.load_model(model_bname)
        self.model = model_handler.model
        model_handler.model = None
        self.model.trainable = False

    def extract_pre_aug_imgs_from_big_imgs(
        self,
        use_model: bool = False,
        min_area_cnt: int = AREA_BORDER - 10,
        expected_pts: int = ARROW_CONTOUR_POINTS,
        ext_imgs_path: Opt[str] = None,
        org_pos_sub_path: Opt[str] = None,
        org_neg_sub_path: Opt[str] = None
    ) -> None:
        self.ext_imgs_path = handle_options(ext_imgs_path, self.ext_imgs_path, RAW_IMGS_PATH)
        self.org_pos_sub_path = handle_options(org_pos_sub_path, self.org_pos_sub_path, ORIGINAL_POS_SUB_PATH)
        self.org_neg_sub_path = handle_options(org_neg_sub_path, self.org_neg_sub_path, ORIGINAL_NEG_SUB_PATH)

        pl.Path(self.org_pos_sub_path).mkdir(parents=True, exist_ok=True)
        pl.Path(self.org_neg_sub_path).mkdir(parents=True, exist_ok=True)

        for img_filename in pl.Path(self.ext_imgs_path).glob('*.jpg'):
            gray_img = cv2.imread(str(img_filename), cv2.IMREAD_GRAYSCALE)
            if gray_img is None:
                print(f'WARNING: {img_filename} could not be loaded')
                continue

            blurred = cv2.blur(gray_img, BLUR_KERNEL)
            cnts = extract_cnts(blurred)

            if use_model and self.model is not None:
                filtered_list, filtered_cnts, cnt_hull_pts_list, hull_rot_pts = filter_cnts(cnts, gray_img, expected_pts)
                if not len(filtered_list):
                    print(f'no candidate for prediction found in {img_filename}')
                    continue

                prediction = self.model.predict(filtered_list).flatten()
                pos_cnts, neg_cnts, _, _, _, _ = sort_cnts(prediction, filtered_cnts, cnt_hull_pts_list, hull_rot_pts)
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
                )
                if pos_cnt is not None:
                    small_img = extract_img_from_cnt(gray_img, pos_cnt)
                    save_img(self.org_pos_sub_path, small_img)

                if neg_cnts is not None:
                    for neg_cnt in neg_cnts:
                        small_img = extract_img_from_cnt(gray_img, neg_cnt)
                        save_img(self.org_neg_sub_path, small_img)

            elif key == Keys.Y:
                if len(pos_cnts) == 1:
                    small_img = extract_img_from_cnt(gray_img, pos_cnts[0])
                    save_img(self.org_pos_sub_path, small_img)

                else:
                    print(f'too many values found in {img_filename} -> no save')

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
    ) -> None:
        self.org_pos_path = handle_options(org_pos_path, self.org_pos_path, ORIGINAL_POS_PATH)
        self.training_arrows_path = handle_options(training_arrows_path, self.training_arrows_path, ARROWS_PATH)
        pl.Path(self.training_arrows_path).mkdir(parents=True, exist_ok=True)
        self.aug_imgs_and_build_dataset(
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
    ) -> None:
        self.org_neg_path = handle_options(org_neg_path, self.org_neg_path, ORIGINAL_NEG_PATH)
        self.training_anything_path = handle_options(training_anything_path, self.training_anything_path, ANYTHING_PATH)
        pl.Path(self.training_anything_path).mkdir(parents=True, exist_ok=True)
        self.aug_imgs_and_build_dataset(
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
    ) -> None:
        sub_paths = list(pl.Path(source_path).glob('*'))
        if len(sub_paths) != 1:
            raise ValueError(f'only one sub path should exist, found {sub_paths} in {source_path}')

        path = pl.Path(sub_paths[0])
        nbr_files = len(list(path.glob('*.jpg')))
        if not nbr_files:
            raise ValueError(f'warn: no jpgs found in {path}')

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
                        raise ValueError(f'retval {retval} saving augmented image: {filename}')
                    num += 1

        print(f'number of created images: {num}')

    def extract_neg_candidates(self) -> None:
        path = pl.Path(BIG_NEG_IMGS_PATH)
        path.mkdir(exist_ok=True)
        pl.Path(UNUSED_NEG_PATH).mkdir(exist_ok=True)

        for num, image_name in enumerate(path.glob('*.jpg')):
            big_neg_image = cv2.imread(str(image_name))
            if big_neg_image is None:
                raise ValueError(f'could not load image {image_name}')

            gray_img = cv2.cvtColor(big_neg_image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.blur(gray_img, BLUR_KERNEL)
            cnts = extract_cnts(blurred)

            for con in cnts:
                small_img = filter_and_extract_img_from_cnt(gray_img, con)
                if small_img is not None:
                    save_img(UNUSED_NEG_PATH, small_img)

            if num % 10 == 0:
                print(f'at least {num} raw images finished', end='\r')

    def move_part_of_unused_to_neg_dataset(self, nbr_imgs: int = ROUGHLY_NBR_OF_FILES_PER_CLASS) -> None:
        pl.Path(ANYTHING_PATH).mkdir(parents=True, exist_ok=True)
        used_neg_filenames = itertools.islice(pl.Path(UNUSED_NEG_PATH).glob('*.jpg'), nbr_imgs)
        for file in used_neg_filenames:
            move(file, PurePath(ANYTHING_PATH, file.name))
