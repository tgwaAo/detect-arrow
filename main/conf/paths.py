from pathlib import PurePath

RAW_VIDS_PATH = '../../raw-positive-videos/'
RAW_IMGS_PATH = '../../raw-positive-images/'
RAW_NEG_IMGS_PATH = '../../raw-negative-images/'
UNUSED_NEG_PATH = '../../unused-negatives/'
ORIGINAL_POS_PATH = '../original-positives/'
ORIGINAL_POS_SUBPATH = str(PurePath(ORIGINAL_POS_PATH, 'original-positives/'))
DATASET_PATH = '../dataset/'
ARROWS_PATH = str(PurePath(DATASET_PATH, 'arrows/'))
ANYTHING_PATH = str(PurePath(DATASET_PATH, 'anything/'))
MODEL_PATH = '../model/'
EXAMPLES_PATH = '../example-images/'
CALIB_IMGS_PATH = '../../calibration-images/'
CAM_CONFIG_PATH = '../../cam-config/'
PRINTED_PATH = '../../printed-values'
PRINTED_MEASUREMENT_FNAME = str(PurePath(PRINTED_PATH, 'calibration_measurements.json'))

