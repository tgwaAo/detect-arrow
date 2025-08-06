from pathlib import PurePath

RAW_VIDS_PATH = '../raw-videos/'
RAW_IMGS_PATH = '../raw-images/'
BIG_NEG_IMGS_PATH = '../big-negative-images/'
UNUSED_NEG_PATH = '../unused-negative-images/'
ORIGINAL_POS_PATH = '../original-positives/'
ORIGINAL_POS_SUB_PATH = str(PurePath(
    ORIGINAL_POS_PATH,
    PurePath(ORIGINAL_POS_PATH).name
))
ORIGINAL_NEG_PATH = '../original-negatives/'
ORIGINAL_NEG_SUB_PATH = str(PurePath(
    ORIGINAL_NEG_PATH, PurePath(ORIGINAL_NEG_PATH).name
))
DATASET_PATH = '../dataset/'
ARROWS_PATH = str(PurePath(DATASET_PATH, 'arrows/'))
ANYTHING_PATH = str(PurePath(DATASET_PATH, 'anything/'))
EXAMPLES_PATH = '../example-images/'
CALIB_IMGS_PATH = '../calibration-images/'
CAM_CONFIG_PATH = '../cam-config/'
CAM_CONFIG_BNAME = 'cam_conf.json'
PRINTED_PATH = '../printed-values'
PRINTED_BNAME = 'coords_of_arrow.txt'
PRINTED_MEASUREMENT_FNAME = str(PurePath(
    PRINTED_PATH,
    'calibration_measurements.json'
))
MODELS_PATH = '../models/'
MODEL_BNAME = 'arrow_detection.keras'
