import pathlib as pl

import numpy as np
import cv2

import numpy.typing as npt
from typing import Optional as Opt
from typing import Union
from typing import TypeVar
from keras.models import Sequential
from collections.abc import Sequence

from conf.imgs import MIN_WIDTH_TO_HEIGHT
from conf.imgs import MAX_WIDTH_TO_HEIGHT
from conf.imgs import SIMILAR_AREA_PERCENT
from conf.imgs import AREA_BORDER
from conf.imgs import TARGET_SIZE
from conf.imgs import ARROW_CONTOUR_POINTS
from datetime import datetime
from datetime import timezone

T = TypeVar('T')
type SeqLike[T] = Union[
    Sequence[T],
    npt.NDArray[T]
]
type CntContainer = Sequence[npt.NDArray[np.integer]]
type ListOfRetvals = list[
    Opt[
        tuple[
            npt.NDArray[np.floating],
            npt.NDArray[np.floating],
            Sequence[npt.NDArray[np.integer]],
            npt.NDArray[np.integer],
            float,
            npt.NDArray[np.floating]
        ]
    ]
]


def get_newest_fname_in_path(path: str) -> str:
    return str(max(pl.Path(path).glob('*'), key=lambda p: p.stat().st_ctime))


def get_current_time_string() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S-%f')


def get_nbr_of_imgs_for_aug(path: str, text: str) -> Opt[int]:
    nbr_files = len(list(pl.Path(path).iterdir()))
    print(f'got {nbr_files} files in {path}')
    ans = input(f'roughly created size for {text} dataset [None] >>')
    if ans.isdigit():
        return int(ans)
    return None


def create_sub_path_with_nbr(path: str, nbr: int) -> str:
    path = pl.PurePath(path)
    return str(path.parent / f'{path.name}-{nbr}')


def handle_options(given: str, existing: str, prepared: str) -> str:
    if given is None:
        if existing is None:  # type: ignore
            return prepared
    else:
        return given
    return existing  # type: ignore


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


def get_user_ans_path(
        paths: list[pl.Path],
        new_candidate: Opt[pl.Path] = None
) -> tuple[Opt[pl.Path], int]:
    nbr_path = -1
    for nbr_path, path in enumerate(paths):
        print(f'{nbr_path}: {path}')

    if new_candidate:
        print('new candidate')
        print(f'{nbr_path + 1}: {new_candidate}')

    ans = input('choose number [default:none] >>')
    if ans.isdigit():
        ans_int = int(ans)
        if ans_int <= nbr_path:
            return paths[ans_int], ans_int
        else:
            return new_candidate, nbr_path + 1
    else:
        raise ValueError('input is not a digit')


def choose_costum_path(
        ref_path: str,
        only_existing: bool = False
) -> tuple[pl.Path, int]:
    ref_path = pl.Path(ref_path)
    paths = srtd_lst_candidates(ref_path)
    if not only_existing:
        nbr_next_path = costum_sort(paths[-1]) + 1
        new_candidate = pl.Path(
            f'{ref_path.parent}', f'{ref_path.name}-{nbr_next_path}'
        )
    else:
        new_candidate = None
    user_ans = get_user_ans_path(paths, new_candidate)
    if user_ans[0] is None:
        raise ValueError('path is not defined')
    return user_ans  # type: ignore


def filter_and_extract_img_from_cnt(
    gray_img: npt.NDArray[np.integer],
    cnt: npt.NDArray[np.integer],
    area_filter: bool = True,
    w_h_filter: bool = True
) -> Opt[npt.NDArray[np.integer]]:
    min_rect = cv2.minAreaRect(cnt)
    center, size, angle = min_rect
    area = size[0] * size[1]

    if area_filter and area < AREA_BORDER:
        return None

    low_value = min(size[0], size[1])
    high_value = max(size[0], size[1])
    width_to_height = low_value / high_value

    if (
        MIN_WIDTH_TO_HEIGHT < width_to_height < MAX_WIDTH_TO_HEIGHT
        or not w_h_filter
    ):
        return extract_img_from_cnt(gray_img, min_rect=min_rect)
    return None


def save_img(path: str, gray_img: npt.NDArray[np.integer]) -> None:
    timestring = get_current_time_string()
    retval = cv2.imwrite(
        str(pl.PurePath(path, f'img_{timestring}.jpg')), gray_img
    )
    if not retval:
        raise IOError(f'could not write image: {retval}')


def extract_img_from_cnt(
    gray_img: npt.NDArray[np.integer],
    cnt: npt.NDArray[np.integer] = None,
    min_rect: tuple[tuple[float, float], tuple[float, float], float] = None
) -> npt.NDArray[np.integer]:
    if min_rect is None:
        if cnt is None:
            raise ValueError('cnt or shape should be given')
        else:
            min_rect = cv2.minAreaRect(cnt)
    cropped_img, _, _ = rotate_and_crop(gray_img, min_rect)
    small_img = cv2.resize(cropped_img, TARGET_SIZE)
    return small_img


def filter_and_extract_norm_img_from_cnt(
    gray_img: npt.NDArray[np.integer],
    cnt: npt.NDArray[np.integer]
) -> Opt[npt.NDArray[np.floating]]:
    small_img = filter_and_extract_img_from_cnt(gray_img, cnt)
    if small_img is not None:
        norm_small_img = small_img / 255
        return norm_small_img

    return None


def extract_cnts(
    img: npt.NDArray[np.integer],
    sigma: float = .33
) -> tuple[npt.NDArray[np.integer]]:
    median_img = np.median(img)
    lower = int(max(0, (1.0 - sigma) * median_img))
    upper = int(min(255, (1.0 + sigma) * median_img))
    thresh_img = cv2.Canny(img, lower, upper)
    cnts, _ = cv2.findContours(
        thresh_img,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
    return cnts


def extract_feature_pts(
    pos_cnt: npt.NDArray[np.integer],
    factors: list[float] = [0.01, 0.015, 0.002, 0.0095, 0.009],
    nbr_expected_pts: int = None
) -> Opt[list[npt.NDArray[np.floating]]]:
    retval = cv2.arcLength(pos_cnt, True)
    for factor in factors:
        hull_pts: npt.NDArray[np.integer] = cv2.approxPolyDP(
            pos_cnt,
            factor * retval,
            True
        )
        merged_hull_pts = merge_points(hull_pts)
        if (
            nbr_expected_pts is not None
            and len(merged_hull_pts) == nbr_expected_pts
        ):
            return merged_hull_pts
        elif nbr_expected_pts is None:
            return merged_hull_pts

    return None


def merge_points(
        pts: npt.NDArray[np.integer],
        max_merge_dist: int = 4
) -> list[npt.NDArray[np.floating]]:
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
            filtered_point = (
                    np.sum(to_merge_bundle, axis=0)
                    / len(to_merge_bundle)
            )
            filtered_points.append(filtered_point)

    return filtered_points


def rotate_and_crop(
        image: npt.NDArray[np.integer],
        min_area_rect: tuple[tuple[float, float], tuple[float, float], float],
        factor: float = 1.3,
        cnt: Opt[npt.NDArray[np.integer]] = None,
        nbr_expected_pts: int = None
) -> tuple[
     npt.NDArray[np.integer],
     Opt[list[npt.NDArray[np.floating]]],
     Opt[npt.NDArray[np.floating]]
]:
    box = cv2.boxPoints(min_area_rect)
    box = np.intp(box)

    width = round(min_area_rect[1][0])
    height = round(min_area_rect[1][1])

    size_of_transformed_image = max(min_area_rect[1])
    min_needed_height = int(np.sqrt(2 * np.power(size_of_transformed_image, 2)))
    width_to_height = min_area_rect[1][0] / min_area_rect[1][1]
    min_rect_angle_deg = get_rotation(min_area_rect[2], width_to_height)
    size = (min_needed_height, min_needed_height)

    x_coordinates_of_box = box[:, 0]  # type: ignore
    y_coordinates_of_box = box[:, 1]  # type: ignore
    x_min = min(x_coordinates_of_box)
    x_max = max(x_coordinates_of_box)
    y_min = min(y_coordinates_of_box)
    y_max = max(y_coordinates_of_box)

    M = cv2.getRotationMatrix2D(
        (size[0] / 2, size[1] / 2),
        min_rect_angle_deg,
        1.0
    )
    center = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
    cropped_image = cv2.getRectSubPix(image, size, center)
    rotated = cv2.warpAffine(cropped_image, M, size)

    if width_to_height >= 1:
        cropped_rot_img = cv2.getRectSubPix(
            rotated,
            patchSize=(int(factor * height), int(factor * width)),
            center=(size[0] / 2, size[1] / 2)
        )
    else:
        cropped_rot_img = cv2.getRectSubPix(
            rotated,
            patchSize=(int(factor * width), int(factor * height)),
            center=(size[0] / 2, size[1] / 2)
        )

    if cnt is not None:
        hull_pts = extract_feature_pts(cnt, nbr_expected_pts=nbr_expected_pts)

        if hull_pts is None:
            rot_pts = None
            return cropped_rot_img, hull_pts, rot_pts

        hull_pts_ = np.array(hull_pts) - min_area_rect[0]
        rot_pts = np.zeros((len(hull_pts_), 2), dtype=float)

        for idx, pt in enumerate(hull_pts_):
            angle_rad = (
                np.arctan2(pt[1], pt[0])
                - np.deg2rad(min_rect_angle_deg)
            )
            dist = np.hypot(pt[0], pt[1])
            pt_x = dist * np.cos(angle_rad)
            pt_y = dist * np.sin(angle_rad)
            rot_pts[idx] = pt_x, pt_y

    else:
        hull_pts = None
        rot_pts = None
    return cropped_rot_img, hull_pts, rot_pts


def get_rotation(min_area_angle: float, width_to_height: float) -> float:
    if width_to_height >= 1:
        min_rect_angle_deg = -1 * (90 - min_area_angle)
    else:
        min_rect_angle_deg = min_area_angle
    return min_rect_angle_deg


def sort_cnts(
    preds: npt.NDArray[np.floating],
    cnts: CntContainer,
    cnt_hull_pts_list: npt.NDArray[np.floating],
    hull_rot_pts: npt.NDArray[np.floating]
) -> tuple[
    Opt[CntContainer],
    Opt[CntContainer],
    Opt[list[float]],
    Opt[list[float]],
    Opt[list[npt.NDArray[np.floating]]],
    Opt[list[npt.NDArray[np.floating]]]
]:
    pos_idxs = np.where(preds >= 0.5)[0]

    if len(pos_idxs):
        pos_cnts = [
            element for idx, element in enumerate(cnts) if idx in pos_idxs
        ]
        pos_preds = [
            element
            for idx, element in enumerate(preds)
            if idx in pos_idxs
        ]
        pos_cnt_hull_pts_list = [
            element
            for idx, element in enumerate(cnt_hull_pts_list)
            if idx in pos_idxs
        ]
        pos_hull_rot_pts = [
            element
            for idx, element in enumerate(hull_rot_pts)
            if idx in pos_idxs
        ]
    else:
        pos_cnts = None
        pos_cnt_hull_pts_list = None
        pos_hull_rot_pts = None
        pos_preds = None

    if len(pos_idxs) < len(preds):
        neg_cnts = [
            element
            for idx, element in enumerate(cnts)
            if idx not in pos_idxs
        ]
        neg_preds = [
            element for idx, element in enumerate(preds) if idx not in pos_idxs
        ]
    else:
        neg_cnts = None
        neg_preds = None

    retval = (
        pos_cnts,
        neg_cnts,
        pos_preds,
        neg_preds,
        pos_cnt_hull_pts_list,
        pos_hull_rot_pts
    )
    return retval


def filter_cnts(
    cnts: CntContainer,
    gray_img: npt.NDArray[np.integer] = None,
    nbr_expected_pts: int = None
) -> tuple[
    npt.NDArray[np.integer],
    list[npt.NDArray[np.integer]],
    npt.NDArray[np.integer],
    npt.NDArray[np.floating]
]:
    # noinspection PyTypeChecker
    small_imgs = []
    # noinspection PyTypeChecker
    filtered_cnts: list[npt.NDArray[np.integer]] = []
    # noinspection PyTypeChecker
    hull_rot_pts = []
    cnt_hull_pts_list = []
    center_list: list[tuple[float, float]] = []
    area_list: list[int] = []
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
                # two contours of same edge
                if percentage > SIMILAR_AREA_PERCENT and too_close:
                    skip = True
                    break

            if skip:
                continue

            if gray_img is not None:
                cropped_img, cnt_hull_pts, rot_pts = rotate_and_crop(
                    gray_img,
                    min_rect,
                    cnt=cnt,
                    nbr_expected_pts=nbr_expected_pts
                )

                if (
                    nbr_expected_pts
                    and rot_pts is not None
                    and len(rot_pts) == nbr_expected_pts
                ):
                    cnt_hull_pts_list.append(cnt_hull_pts)
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
    cnt_hull_pts_list = np.array(cnt_hull_pts_list)
    hull_rot_pts = np.array(hull_rot_pts)
    return small_imgs, filtered_cnts, cnt_hull_pts_list, hull_rot_pts


def more_pts_up(
    pts: Sequence[Sequence[float, float]] | npt.NDArray[np.floating],  # type: ignore
    center: Sequence[int] = None
) -> bool:
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


def angle_x_axis(pt: SeqLike[np.floating]) -> float:
    angle_rad = np.arctan2(pt[0], pt[1])
    if angle_rad < 0:
        angle_rad += 2 * np.pi
    return angle_rad


def get_max_y_dist_reference(
    pts: Sequence[Sequence[float]] | npt.NDArray[np.floating],
    ref_up: bool
) -> Sequence[float] | npt.NDArray[np.floating]:
    comp_dist_y = 0.
    ref_pt = None

    for pt in pts:
        y_pt = pt[1]
        if ref_up and y_pt < comp_dist_y:
            comp_dist_y = y_pt
            ref_pt = pt
        elif not ref_up and comp_dist_y < y_pt:
            comp_dist_y = y_pt
            ref_pt = pt

    if ref_pt is None:
        raise ValueError('could not find a valid reference point')
    return ref_pt


def rot_centered_pts(
        pts: Sequence[Sequence[float]] | npt.NDArray[np.floating],
        ref_angle_rad: float
) -> npt.NDArray[np.floating]:
    rot_pts = np.zeros((len(pts), 2), dtype=float)
    for idx, pt in enumerate(pts):
        angle_rad = np.arctan2(pt[1], pt[0]) + ref_angle_rad
        dist = np.hypot(pt[0], pt[1])
        pt_x = np.round(dist * np.cos(angle_rad), decimals=4)
        pt_y = np.round(dist * np.sin(angle_rad), decimals=4)
        rot_pts[idx] = pt_x, pt_y

    return rot_pts


def sort_pts_by_angles(
    rot_pts: Sequence[Sequence[float]] | npt.NDArray[np.floating],
    org_pts: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    angles_rad = [angle_x_axis(pt) for pt in rot_pts]  # type: ignore
    idx_sorted = np.argsort(angles_rad)
    return org_pts[idx_sorted]


def sort_pt_biggest_dist_y(
    pts: Sequence[Sequence[float]] | npt.NDArray[np.floating],
    ref_up: bool,
    org_pts: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    ref_pt = get_max_y_dist_reference(pts, ref_up)
    ref_angle_rad = (np.pi / 2) - np.arctan2(ref_pt[1], ref_pt[0])
    rot_pts = rot_centered_pts(pts, ref_angle_rad)
    return sort_pts_by_angles(rot_pts, org_pts)


def calc_rot_and_trans(
    homogr: npt.NDArray[np.floating],
    mtx: npt.NDArray[np.floating]
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
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
    beta = np.rad2deg(np.arctan2(
        -rot[2, 0],
        np.sqrt(rot[2, 1] * rot[2, 1] + rot[2, 2] * rot[2, 2])
    ))
    gamma = np.rad2deg((np.arctan2(rot[1, 0], rot[0, 0])))
    return np.array((alpha, beta, gamma)), trans


def est_cnts_in_img(
    gray_img: npt.NDArray[np.integer],
    model: Sequential,
    verbose: bool = False,
    expected_pts: int = ARROW_CONTOUR_POINTS
) -> Opt[
    tuple[
        Opt[CntContainer],
        Opt[CntContainer],
        Opt[list[float]],
        Opt[list[float]],
        Opt[list[npt.NDArray[np.floating]]],
        Opt[list[npt.NDArray[np.floating]]]
    ]
]:
    cnts = extract_cnts(gray_img)

    filtered_images, cnts, cnt_hull_pts_list, hull_rot_pts = filter_cnts(
        cnts,
        gray_img,
        expected_pts
    )

    if not len(filtered_images):
        if verbose:
            print('no candidate for prediction found')
        return None

    prediction = model(filtered_images).numpy().flatten()

    # noinspection PyTypeChecker
    return sort_cnts(
        prediction,
        cnts,
        cnt_hull_pts_list,  # type: ignore
        hull_rot_pts
    )


def est_pose_of_cnt(
    points_printed: npt.NDArray[np.integer],
    cnt_hull_pts: npt.NDArray[np.floating],
    hull_rot_pts: npt.NDArray[np.floating],
    mtx: npt.NDArray[np.floating],
    verbose: bool = False
) -> Opt[tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating]
]]:
    if len(cnt_hull_pts) != ARROW_CONTOUR_POINTS:
        if verbose:
            print(
                f'incorrect number of hull points,'
                f'got {len(cnt_hull_pts)} need {ARROW_CONTOUR_POINTS}'
            )
        return None
    cnt_hull_pts = np.reshape(cnt_hull_pts, (ARROW_CONTOUR_POINTS, 2))
    ref_up = more_pts_up(hull_rot_pts)
    cnt_hull_pts = sort_pt_biggest_dist_y(hull_rot_pts, ref_up, cnt_hull_pts)
    homogr, _ = cv2.findHomography(points_printed, cnt_hull_pts, cv2.RANSAC)
    rot, trans = calc_rot_and_trans(homogr, mtx)
    return rot, trans, cnt_hull_pts


def est_pose_in_img(
    gray_img: npt.NDArray[np.integer],
    model: Sequential,
    points_printed: npt.NDArray[np.integer],
    mtx: npt.NDArray[np.floating],
    verbose: bool = False
) -> Opt[tuple[
    Opt[npt.NDArray[np.floating]],
    Opt[npt.NDArray[np.floating]],
    npt.NDArray[np.integer],
    float,
    Opt[npt.NDArray[np.floating]]
]]:
    cnt_result = est_cnts_in_img(gray_img, model, verbose=verbose)
    if cnt_result is None:
        return None

    (
        pos_cnts,
        neg_cnts,
        pos_preds,
        neg_preds,
        cnt_hull_pts_list,
        hull_rot_pts
    ) = cnt_result

    if [  # check for any positive value
        x
        for x in (pos_cnts, cnt_hull_pts_list, hull_rot_pts, pos_preds)
        if x is None
    ]:
        if verbose:
            print('no positive contour found')
        # noinspection PyTypeChecker
        cnt_result = None

    if cnt_result is None:
        return None

    # noinspection PyTypeChecker
    idx = np.argmax(pos_preds)  # type: ignore
    pos_pred = pos_preds[idx]  # type: ignore
    best_cnt = pos_cnts[idx]  # type: ignore
    best_cnt_hull_pts = cnt_hull_pts_list[idx]  # type: ignore
    best_hull_rot_pts = hull_rot_pts[idx]  # type: ignore
    result = est_pose_of_cnt(
        points_printed,
        best_cnt_hull_pts,
        best_hull_rot_pts,
        mtx,
        verbose
    )

    if result is None:
        return None, None, best_cnt, pos_pred, None

    R, T, sorted_hull_pts = result
    # noinspection PyTypeChecker
    return R, T, best_cnt, pos_pred, sorted_hull_pts


def est_poses_in_img(
    gray_img: npt.NDArray[np.integer],
    model: Sequential,
    points_printed: npt.NDArray[np.integer],
    mtx: npt.NDArray[np.floating],
    verbose: bool = False
) -> Opt[list[tuple[
    Opt[npt.NDArray[np.floating]],
    Opt[npt.NDArray[np.floating]],
    npt.NDArray[np.integer],
    np.float32,
    Opt[npt.NDArray[np.floating]]
]]]:
    cnt_result = est_cnts_in_img(
        gray_img,
        model,
        verbose=verbose
    )
    if cnt_result is None:
        return None

    (
        pos_cnts,
        neg_cnts,
        pos_preds,
        neg_preds,
        cnt_hull_pts_list,
        hull_rot_pts
    ) = cnt_result

    if [  # check for any positive value
        x
        for x in (pos_cnts, cnt_hull_pts_list, hull_rot_pts, pos_preds)
        if x is None
    ]:
        if verbose:
            print('no positive contour found')
        return None

    all_ret_vals: ListOfRetvals = [None] * len(pos_cnts)  # type: ignore
    for idx in range(len(pos_cnts)):  # type: ignore
        pose_result = est_pose_of_cnt(
            points_printed,
            cnt_hull_pts_list[idx],  # type: ignore
            hull_rot_pts[idx],  # type: ignore
            mtx,
            verbose
        )
        if pose_result is None:
            all_ret_vals[idx] = (  # type: ignore
                None,
                None,
                pos_cnts[idx],  # type: ignore
                pos_preds[idx],  # type: ignore
                None
            )
            if verbose:
                print(f'could not estimate pose at {idx}')
        else:
            R, T, sorted_hull_pts = pose_result
            # noinspection PyTypeChecker
            all_ret_vals[idx] = (  # type: ignore
                R,
                T,
                pos_cnts[idx],  # type: ignore
                pos_preds[idx],  # type: ignore
                sorted_hull_pts
            )

    return all_ret_vals  # type: ignore


def save_points_in_img(
    pts: npt.NDArray[np.floating],
    offset: tuple[int, int] = (200, 200),
    name: str = 'test.jpg'
) -> None:
    black = np.zeros((480, 640, 3))
    pts = pts.astype(int) + offset
    for idx, pt in enumerate(pts):
        cv2.circle(black, pt, 3, (255,255,255), -1)
        cv2.putText(
            black,
            str(idx),
            pt + (5, -5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255,255,255)
        )

    cv2.imwrite(name, black)
