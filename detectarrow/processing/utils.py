import pathlib as pl

import numpy as np
import cv2

import numpy.typing as npt
from typing import Optional as Opt, Any
from typing import TypeAlias
from typing import Union
from keras.models import Sequential
from numpy import ndarray, dtype
from tensorflow.python.framework.ops import EagerTensor
from typing import Union
import tensorflow as tf

from detectarrow.conf.paths import RAW_VIDS_PATH
from detectarrow.conf.paths import RAW_POS_IMGS_PATH
from detectarrow.conf.imgs import MIN_WIDTH_TO_HEIGHT
from detectarrow.conf.imgs import MAX_WIDTH_TO_HEIGHT
from detectarrow.conf.imgs import SIMILAR_AREA_PERCENT
from detectarrow.conf.imgs import AREA_BORDER
from detectarrow.conf.imgs import TARGET_SIZE
from detectarrow.conf.imgs import COMPARED_SIZE
from detectarrow.conf.imgs import ARROW_CONTOUR_POINTS
from detectarrow.conf.imgs import BLUR_KERNEL
from datetime import datetime
from datetime import timezone

type cnt_container = Union[
    list[npt.NDArray[int], ...],
    tuple[npt.NDArray[int], ...]
]


def get_newest_fname_in_path(path: str) -> str:
    return str(max(pl.Path(path).glob('*'), key=lambda p: p.stat().st_ctime))


def get_current_time_string() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S-%f')


def create_sub_path_with_nbr(path: str, nbr: int) -> str:
    path = pl.PurePath(path)
    return str(path.parent / f'{path.name}-{nbr}')


def handle_options(given: str, existing: str, prepared: str) -> str:
    if given is None:
        if existing is None:
            return prepared
    else:
        return given
    return existing


def costum_sort(element: pl.PurePath) -> int:
    sep_words = str(element).split('-')
    if sep_words[-1].isdigit():
        return int(sep_words[-1])
    else:
        return 0


def srtd_lst_candidates(ref_path: pl.Path) -> list[pl.Path]:
    basename = ref_path.name
    paths = sorted(ref_path.parent.glob(f'{basename}*'), key=costum_sort)
    return paths


def get_user_ans(paths: list[pl.Path], new_candidate=None) -> Opt[tuple[pl.Path, int]]:
    idx = -1
    for idx, path in enumerate(paths):
        print(f'{idx}: {path}')

    if new_candidate:
        print('new candidate')
        print(f'{idx + 1}: {new_candidate}')

    ans = input('choose number [default:none] >>')
    if ans.isdigit():
        ans = int(ans)
        if ans <= idx:
            return paths[ans], ans
        else:
            return new_candidate, idx + 1

    else:
        return None


def choose_costum_path(ref_path: str, only_existing: bool = False) -> Opt[tuple[pl.Path, int]]:
    ref_path = pl.Path(ref_path)
    paths = srtd_lst_candidates(ref_path)
    if not only_existing:
        nbr_next_path = costum_sort(paths[-1]) + 1
        new_candidate = pl.Path(f'{ref_path.parent}', f'{ref_path.name}-{nbr_next_path}')
    else:
        new_candidate = None

    return get_user_ans(paths, new_candidate)


def filter_and_extract_img_from_cnt(
    gray_img: npt.NDArray[np.uint8],
    cnt: npt.NDArray[int],
    area_filter: bool = True,
    w_h_filter: bool = True
) -> Opt[npt.NDArray[np.uint8]]:
    min_rect = cv2.minAreaRect(cnt)
    center, size, angle = min_rect
    area = size[0] * size[1]

    if area_filter and area < AREA_BORDER:
        return None

    low_value = min(size[0], size[1])
    high_value = max(size[0], size[1])
    width_to_height = low_value / high_value

    if MIN_WIDTH_TO_HEIGHT < width_to_height < MAX_WIDTH_TO_HEIGHT or not w_h_filter:
        return extract_img_from_cnt(gray_img, min_rect)

    return None


def save_img(path: str, gray_img: npt.NDArray[np.uint8]) -> None:
    timestring = get_current_time_string()
    retval = cv2.imwrite(str(pl.PurePath(path, f'img_{timestring}.jpg')), gray_img)
    if not retval:
        raise ValueError(f'could not write image: {retval}')


def extract_img_from_cnt(
    gray_img: npt.NDArray[np.uint8],
    shape: npt.NDArray[int] | tuple[tuple[float], tuple[float], float]
) -> npt.NDArray[np.uint8]:
    if not isinstance(shape, tuple):
        shape = cv2.minAreaRect(shape)
    cropped_img, _ = rotate_and_crop(gray_img, shape)
    small_img = cv2.resize(cropped_img, TARGET_SIZE)
    return small_img


def filter_and_extract_norm_img_from_cnt(
    gray_img: npt.NDArray[np.uint8],
    cnt: npt.NDArray[int]
) -> Opt[npt.NDArray[np.uint8]]:
    small_img = filter_and_extract_img_from_cnt(gray_img, cnt)
    if small_img is not None:
        small_img = small_img / 255
        return small_img

    return None


def extract_cnts(img: npt.NDArray[np.uint8], sigma: float = .33) -> cnt_container:
    v = np.median(img)
    # ---- apply automatic Canny edge detection using the computed median----
    lower = int(max(0, (1.0 - sigma) * v))  # ---- lower threshold
    upper = int(min(255, (1.0 + sigma) * v))  # ---- upper threshold
    thresh_img = cv2.Canny(img, lower, upper)
    cnts, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cnts


def extract_feature_pts(pos_cnt: npt.NDArray[int], factor: float = 0.01) -> npt.NDArray[int]:
    retval = cv2.arcLength(pos_cnt, True)
    return cv2.approxPolyDP(pos_cnt, factor * retval, True)


def merge_points(pts: npt.NDArray[int], max_merge_dist: int = 4) -> list[npt.NDArray[int | float]]:
    to_merge = []
    checked_points_idx = []
    last_to_merge = False
    for idx in range(len(pts) - 1):
        first_point = pts[idx, 0]
        if idx not in checked_points_idx:
            to_merge_bundle = [first_point]

            for idx2 in range(idx + 1, len(pts)):
                second_point = pts[idx2, 0]
                dist = np.abs(first_point - second_point)

                if dist[0] < max_merge_dist and dist[1] < max_merge_dist:
                    to_merge_bundle.append(second_point)
                    checked_points_idx.append(idx2)
                    if idx2 == len(pts) - 1:
                        last_to_merge = True

            to_merge.append(to_merge_bundle)

    if not last_to_merge:
        to_merge.append([pts[-1, 0]])

    filtered_points = []
    for to_merge_bundle in to_merge:
        if len(to_merge_bundle) == 1:
            filtered_points.append(to_merge_bundle[0])
        else:
            filtered_point = np.sum(to_merge_bundle, axis=0) / len(to_merge_bundle)
            filtered_points.append(filtered_point)

    return filtered_points


def rotate_and_crop(
        image: npt.NDArray[np.uint8],
        min_area_rect: tuple[tuple[float], tuple[float], float],
        factor: float = 1.3,
        cnt: Opt[npt.NDArray[int]] = None
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[float]]:
    box = cv2.boxPoints(min_area_rect)
    box = np.intp(box)

    width = round(min_area_rect[1][0])
    height = round(min_area_rect[1][1])

    size_of_transformed_image = max(min_area_rect[1])
    min_needed_height = int(np.sqrt(2 * np.power(size_of_transformed_image, 2)))

    width_to_height = min_area_rect[1][0] / min_area_rect[1][1]

    min_rect_angle_deg = get_rotation(min_area_rect[2], width_to_height)

    size = (min_needed_height, min_needed_height)

    x_coordinates_of_box = box[:, 0]
    y_coordinates_of_box = box[:, 1]
    x_min = min(x_coordinates_of_box)
    x_max = max(x_coordinates_of_box)
    y_min = min(y_coordinates_of_box)
    y_max = max(y_coordinates_of_box)

    center = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
    cropped = cv2.getRectSubPix(image, size, center)
    M = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), min_rect_angle_deg, 1.0)
    cropped = cv2.warpAffine(cropped, M, size)

    if width_to_height >= 1:
        cropped_rotated = cv2.getRectSubPix(cropped, (int(factor * height), int(factor * width)),
                                            (size[0] / 2, size[1] / 2))
    else:
        cropped_rotated = cv2.getRectSubPix(cropped, (int(factor * width), int(factor * height)),
                                            (size[0] / 2, size[1] / 2))

    if cnt is not None:
        hull_pts = extract_feature_pts(cnt)
        hull_pts = np.array(merge_points(hull_pts)) - min_area_rect[0]
        rot_pts = np.zeros((len(hull_pts), 2))

        for idx, pt in enumerate(hull_pts):
            angle_rad = np.arctan2(pt[1], pt[0]) - np.deg2rad(min_rect_angle_deg)
            dist = np.hypot(pt[0], pt[1])
            pt_x = dist * np.cos(angle_rad)
            pt_y = dist * np.sin(angle_rad)
            rot_pts[idx] = pt_x, pt_y

    else:
        rot_pts = None

    return cropped_rotated, rot_pts


def get_rotation(min_area_angle: float, width_to_height: float) -> float:
    if width_to_height >= 1:
        min_rect_angle_deg = -1 * (90 - min_area_angle)
    else:
        min_rect_angle_deg = min_area_angle
    return min_rect_angle_deg


def sort_cnts(
    prediction: EagerTensor,
    cnts: cnt_container,
    hull_rot_pts: npt.NDArray[float]
) -> tuple[
    Opt[list[int]],
    Opt[list[int]],
    Opt[list[tf.Tensor]],
    Opt[list[tf.Tensor]],
    Opt[list[float]]
]:
    idxs = np.where(prediction >= 0.5)[0]

    if len(idxs):
        positive_contours = [element for idx, element in enumerate(cnts) if idx in idxs]
        pos_prediction = [element for idx, element in enumerate(prediction) if idx in idxs]
        hull_rot_pts = [element for idx, element in enumerate(hull_rot_pts) if idx in idxs]
    else:
        positive_contours = None
        hull_rot_pts = None
        pos_prediction = None

    if len(idxs) < len(prediction):
        negative_contours = [element for idx, element in enumerate(cnts) if idx not in idxs]
        neg_prediction = [element for idx, element in enumerate(prediction) if idx not in idxs]
    else:
        negative_contours = None
        neg_prediction = None

    return positive_contours, negative_contours, pos_prediction, neg_prediction, hull_rot_pts


def filter_cnts(
    cnts: cnt_container,
    gray_img: npt.NDArray[np.uint8] = None,
    expected_pts: int = None
) -> tuple[npt.NDArray[np.uint8], cnt_container, npt.NDArray[float]]:
    # noinspection PyTypeChecker
    small_imgs = []
    # noinspection PyTypeChecker
    filtered_cnts: cnt_container = []
    # noinspection PyTypeChecker
    hull_rot_pts = []
    center_list = []
    area_list = []
    for cnt in cnts:
        min_rect = cv2.minAreaRect(cnt)
        center, size, angle_deg = min_rect
        area = size[0] * size[1]
        if area < AREA_BORDER:
            continue

        low_value = min(size[0], size[1])
        high_value = max(size[0], size[1])
        if high_value == 0:
            continue

        width_to_height = low_value / high_value
        if MIN_WIDTH_TO_HEIGHT < width_to_height < MAX_WIDTH_TO_HEIGHT:
            skip = False
            for c_point, a_point in zip(center_list, area_list):
                too_close = np.all(np.isclose(center, c_point, rtol=0, atol=20))
                low_value = min(area, a_point)
                high_value = max(area, a_point)
                percentage = low_value / high_value
                if percentage > SIMILAR_AREA_PERCENT and too_close:  # two contours of same edge
                    skip = True
                    break

            if skip:
                continue

            if gray_img is not None:
                cropped_img, rot_pts = rotate_and_crop(gray_img, min_rect, cnt=cnt)
                if expected_pts and len(rot_pts) == expected_pts:
                    hull_rot_pts.append(rot_pts)
                    small_img = cv2.resize(cropped_img, TARGET_SIZE)
                    small_imgs.append(small_img)
                    filtered_cnts.append(cnt)
                    area_list.append(area)
                    center_list.append(center)

            else:
                filtered_cnts.append(cnt)
                area_list.append(area)
                center_list.append(center)

    small_imgs = np.array(small_imgs)
    hull_rot_pts = np.array(hull_rot_pts)
    return small_imgs, filtered_cnts, hull_rot_pts


def more_pts_up(pts: npt.NDArray[int | float], center: int = None) -> bool:
    if center is not None:
        y_max = center[1]
    else:
        y_max = 0

    pts_up = 0
    pts_down = 0

    for x, y in pts:
        if y < y_max:
            pts_up += 1
        else:
            pts_down += 1

    return pts_down < pts_up


def angle_x_axis(pt: npt.ArrayLike) -> float:
    angle_rad = np.arctan2(pt[0], pt[1])
    if angle_rad < 0:
        angle_rad += 2 * np.pi
    return angle_rad


def get_max_dist_reference(pts: npt.NDArray, ref_up: bool) -> Opt[npt.NDArray]:
    max_dist = 0
    min_idx = -1
    for idx, pt in enumerate(pts):
        dist = np.power(pt[0], 2) + np.power(pt[1], 2)
        if max_dist < dist:
            if ref_up and pt[1] < 0:  # up is lower
                min_idx = idx
                max_dist = dist
            elif not ref_up and 0 < pt[1]:
                min_idx = idx
                max_dist = dist

    if min_idx == -1:
        return None
    return pts[min_idx]


def rot_centered_pts(pts: npt.NDArray, ref_angle_rad: float) -> npt.NDArray[float]:
    rot_pts = np.zeros((len(pts), 2))
    for idx, pt in enumerate(pts):
        angle_rad = np.arctan2(pt[1], pt[0]) + ref_angle_rad
        dist = np.hypot(pt[0], pt[1])
        pt_x = np.round(dist * np.cos(angle_rad), decimals=4)
        pt_y = np.round(dist * np.sin(angle_rad), decimals=4)
        rot_pts[idx] = pt_x, pt_y

    return rot_pts


def sort_pts_by_angles(
    rot_pts: npt.NDArray,
    org_pts: npt.NDArray
) -> npt.NDArray:
    angles_rad = [angle_x_axis(pt) for pt in rot_pts]
    idx_sorted = np.argsort(angles_rad)
    return org_pts[idx_sorted]


def sort_pt_biggest_dist_center(
    pts: npt.NDArray,
    ref_up: bool,
    org_pts: npt.NDArray
) -> npt.NDArray:
    closest_y = get_max_dist_reference(pts, ref_up)
    ref_angle_rad = (np.pi / 2) - np.arctan2(closest_y[1], closest_y[0])
    rot_pts = rot_centered_pts(pts, ref_angle_rad)
    return sort_pts_by_angles(rot_pts, org_pts)


def calc_rot_and_trans(
    homogr: npt.NDArray,
    mtx: npt.NDArray[float]
) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
    homogr = homogr.T
    h1 = homogr[0]
    h2 = homogr[1]
    h3 = homogr[2]
    mtx_inv = np.linalg.inv(mtx)
    L = 1 / np.linalg.norm(np.dot(mtx_inv, h1))
    r1 = L * np.dot(mtx_inv, h1)
    r2 = L * np.dot(mtx_inv, h2)
    r3 = np.cross(r1, r2)

    trans = L * np.dot(mtx_inv, h3)

    rot = np.array([[r1], [r2], [r3]])
    rot = np.reshape(rot, (3, 3))
    U, S, V = np.linalg.svd(rot, full_matrices=True)
    U = np.matrix(U)
    V = np.matrix(V)
    rot = U * V

    alpha = np.rad2deg(np.arctan2(rot[2, 1], rot[2, 2]))
    beta = np.rad2deg(np.arctan2(-rot[2, 0], np.sqrt(rot[2, 1] * rot[2, 1] + rot[2, 2] * rot[2, 2])))
    gamma = np.rad2deg((np.arctan2(rot[1, 0], rot[0, 0])))
    return np.array((alpha, beta, gamma)), trans


def est_cnts_in_img(
    gray_img: npt.NDArray[np.uint8],
    model: Sequential,
    verbose: bool = False,
    expected_pts: int = ARROW_CONTOUR_POINTS
) -> Opt[
    tuple[
        Opt[npt.NDArray[int]],
        Opt[npt.NDArray[int]],
        Opt[list[tf.Tensor]],
        Opt[list[tf.Tensor]],
        Opt[npt.NDArray[int]]]
]:
    cnts = extract_cnts(gray_img)

    filtered_images, cnts, hull_rot_pts = filter_cnts(cnts, gray_img, expected_pts)

    if not len(filtered_images):
        if verbose:
            print('no candidate for prediction found')
        return None

    prediction = model(filtered_images)

    return sort_cnts(prediction, cnts, hull_rot_pts)


def est_pose_of_cnt(
    best_cnt,
    points_printed,
    best_hull_rot_pts,
    mtx,
    verbose=False
) -> Opt[tuple[npt.NDArray[float], npt.NDArray[float]]]:
    hull_points = extract_feature_pts(best_cnt)
    hull_points = merge_points(hull_points)
    if len(hull_points) != ARROW_CONTOUR_POINTS:
        if verbose:
            print(f'incorrect number of hull points, got {len(hull_points)} need {ARROW_CONTOUR_POINTS}')
        return None
    hull_points = np.reshape(hull_points, (ARROW_CONTOUR_POINTS, 2))
    ref_up = more_pts_up(best_hull_rot_pts)
    hull_points = sort_pt_biggest_dist_center(best_hull_rot_pts, ref_up, hull_points)

    points_printed = sort_pt_biggest_dist_center(points_printed, False, points_printed)
    homogr, _ = cv2.findHomography(points_printed, hull_points, cv2.RANSAC)
    rot, trans = calc_rot_and_trans(homogr, mtx)
    return rot, trans


def est_pose_in_img(
    gray_img: npt.NDArray[np.uint8],
    model: Sequential,
    points_printed: npt.NDArray[np.uint8],
    mtx,
    verbose: bool = False
) -> Opt[list[tuple[npt.NDArray[float], npt.NDArray[float], npt.NDArray[int], float]]]:
    cnt_result = est_cnts_in_img(gray_img, model, verbose=verbose)
    if cnt_result is None:
        return None

    pos_cnts, neg_cnts, pos_preds, neg_preds, hull_rot_pts = cnt_result

    if pos_cnts is None:
        if verbose:
            print('no positive contour found')
        # noinspection PyTypeChecker
        cnt_result = None

    if cnt_result is None:
        return None

    # noinspection PyTypeChecker
    idx = np.argmax(pos_preds)  # list of ensors containing a single float being treated like floats
    best_cnt: npt.NDArray[int] = pos_cnts[idx]
    best_hull_rot_pts = hull_rot_pts[idx]
    result: Opt[tuple[npt.NDArray[float], npt.NDArray[float]]] = est_pose_of_cnt(best_cnt, points_printed, best_hull_rot_pts, mtx, verbose)
    if result is None:
        return None
    R, T = result
    # noinspection PyTypeChecker
    pos_pred: float = pos_preds[idx][0]  # list of ensors containing a single float returns float
    return [(R, T, best_cnt, pos_pred)]


def est_poses_in_img(
    gray_img,
    model,
    points_printed,
    mtx,
    verbose: bool = False
) -> Opt[list[tuple[npt.NDArray[float], npt.NDArray[float], npt.NDArray[int], float]]]:
    cnt_result = est_cnts_in_img(gray_img, model, verbose=verbose)
    if cnt_result is None:
        return None
    pos_cnts, neg_cnts, pos_preds, neg_preds, hull_rot_pts = cnt_result

    if not len(pos_cnts):
        if verbose:
            print('no positive contour found')
        return None

    all_ret_vals: list[
        Opt[tuple[npt.NDArray[float], npt.NDArray[float], npt.NDArray[int], float]]
    ] = [None] * len(pos_cnts)
    for idx in range(len(pos_cnts)):
        cnt_result = est_pose_of_cnt(pos_cnts[idx], points_printed, hull_rot_pts[idx], mtx, verbose)
        if cnt_result is None:
            if verbose:
                print(f'could not estimate pose at {idx}')
        else:
            R, T = cnt_result
            # noinspection PyTypeChecker
            all_ret_vals[idx] = R, T, pos_cnts[idx], pos_preds[idx][0]

    return all_ret_vals



