# detect-arrow

The project is designed to detect the rotation and translation of a printed arrow. It is mostly computer vision based and only uses a very small neural network.

## Paths

### Complete path overview (without files)

- **big-negative-images**  
- **calibration-images**  
- cam-config  
- dataset  
    - anything  
    - arrows  
- detectarrow  
    - conf  
    - inout  
    - processing  
- example-images  
- models  
- original-negatives-1  
    - original-negatives-1  
- original-negatives-2  
    - original-negatives-2  
- original-positives  
    - original-positives  
- original-positives-1  
    - original-positives-1  
- printed-arrow  
- printed-values  
- **raw-images**  
- **raw-images-1**  
- **raw-videos**  
- **raw-videos-1**  
- **unused-negative-images**  

### Path after git clone (without files)

- cam-config  
- dataset  
    - anything  
    - arrows  
- detectarrow  
    - conf  
    - inout  
    - processing  
- example-images  
- models  
- original-negatives-1  
    - original-negatives-1  
- original-negatives-2  
    - original-negatives-2  
- original-positives  
    - original-positives  
- original-positives-1  
    - original-positives-1  
- printed-arrow  
- printed-values  

#### Explanation of cam-config
This is where the camera configuration is stored.  

#### Explanation of dataset
This is where the dataset for training and validation/testing is stored. The files should be added automatically in the extraction process.

#### Explanation of anything
This is where the negative part of the dataset for training and validation/testing is stored. The files should be added automatically in the extraction process.

#### Explanation of arrows
This is where the positive part of the dataset for training and validation/testing is stored. The files should be added automatically in the extraction process.

#### Explanation of detectarrow
This is where the code for this project is located.  

#### Explanation of example-images
This is where some examples for the usage of this project are located.  

#### Explanation of original-negatives*
This is where original negative images for the dataset are stored. These files are not augmented, but might be used for augmentation. This path will be created automatically.

#### Explanation of original-positives*
This is where original positive images for the dataset are stored. These files are not augmented, but might be used for augmentation. This path will be created automatically.

#### Explanation of printed-arrow
This is where the image for the used arrow is located. Feels free to print it and to test the project with your own images and camera.  

#### Explanation of printed-values
This is where files for the callibration and for the estimation of the arrow are located. This files might need to be adjusted for your calibration and estimation.  

### Paths not existend after using git clone

- big-negative-images  
- calibration-images  
- raw-images  
- raw-images-*  
- raw-videos  
- raw-videos-*  
- unused-negative-images  

#### Explanation of big-negative-images

This paths has to be created manually if someone wants to recreate the whole process of this project.
Some big images without the searched object (e.g. arrow) can be put there for an extraction of small negative images later.

#### Explanation of calibration-images

This paths has to be created manually if someone wants to calibrate their own camera.
The details are explained under section "**Calibration**".

#### Explanation of raw-videos

This paths has to be created manually for the extraction of positives and negatives.
Place video files supported by opencv (webm, etc.) and use these files to extract positives and negatives.

#### Explanation of raw-videos-*

This paths has to be created manually if someone wants to repeat the process of raw-videos, but with different data.
Instead of an asterisk, please use a number as reference.

#### Explanation of raw-images

This paths will be created automatically and will be used to store the extracted files of raw-videos.

#### Explanation of raw-images-*

This paths will be created automatically and will be used to store the extracted files of a raw-videos-\*. Typically the number replacing the asterisk will be the same as the one of raw-videos-\*.

#### Explanation of unused-negative-images

This paths will be created automatically and will be used to store the unused extracted files of big-negative-images (there will be many extracted, but the goal is to use only 10 000).

## Usage

### Quick test

1. Install conda
2. Create environment with conda:  
  `conda env create -f environment.yml`
3. Execute  
  `python3 04_estimate_transformation_in_image.py`
4. Look at the images with their estimation
5. Close the output using letter "q"

### Calibration

To get proper estimation result you need to calibrate your camera first (1)(2):  
  
1. You can download and print a chessboard pattern like [https://github.com/opencv/opencv/blob/4.x/doc/pattern.png](https://github.com/opencv/opencv/blob/4.x/doc/pattern.png)  
2. Create a folder/directory "calibration-images"  
3. Make multiple photos of the chessboard pattern from different perspectives and place them inside "calibration-images".  
4. You have to check the config at printed-values/calibration_measurements.json. The distance between points is what it sounds like and the width and height refer to the number of inner corners in width and height. (3)  
5. Run the calibration:  
  `python3 03_calibrate_camera.py`
6. Look at the output  
7. Look at the estimated error on your console  
8. Close the output using letter "q"  

### Use your own images

1. Setup project as described in "Quick test"
2. Calibrate your camera as described in "Calibration"
3. Copy file "04_estimate_transformation_in_image.py" and change the string "EXAMPLES_PATH" in line 44 to your path ( you might need to change the other paths if your copy is created somewhere else).
4. Run the estimation:  
  `python3 <path_to_your_file>`
5. Look at the images with their estimation
6. Close the output using letter "q"

### Use your own camera

1. Setup and calibrate as described in 1. and 2. of "Use your own images"
2. Edit the file cam-config/cam_conf.json.  
3. Start "05_estimate_transformation_in_camera.py":  
  `python3 05_estimate_transformation_in_camera.py`
4. Close the output using letter "q"

### Start the complete process anew

1. Delete "original-positives" and its content
2. Delete "dataset" and its content
3. You might want to change a few names like the path name "arrows" in paths.py
4. Create the directories/folders with the names "raw-videos" and "big-negative-images"
5. Make videos with your searched object and place them in "raw-videos"
6. Collect (big) negative images and place them in "big-negative-images"
7. Setup project as described in "Quick test"
8. Start the preparation:  
  `python3 01_extract_raw_data.py`
9. Look at each image. Press space to ignore the current image and press "n" to extract the positive image manually.  
  In manual search you see a marked contour that wraps around a shape.  
  If you want to ignore the current contour, press space. 
  If you want to want to mark the current contour as the searched object, press "y".  
  You do not have to do it here, but you can also press "n" to mark a contour as negative and extract a negative image.  
  The manual extraction of an image end as soon as the positive contour is found. Then the next image is displayed  
  if there is an image left.
10. The augmentation of the positive files and an extraction of the negative files is done now.
11. Look at the output. There should be a "original-positives" in another "original-positives" with the original content of the extraction.  
  There should also be a "unused-negative-images" with the unused extracted negative images. These can be deleted or manually added to the dataset.  
  The later used dataset is found in "dataset".

### Improve neural network by semiautomatically extracting positives and negatives

If you have issues with the detection of positives, you can improve the detection using this part of the project:

1. Make videos containing the searched object and with the used camera and place them in a newly created raw-videos-<number\> directory/folder 
2. Activate virtualenv
3. Start the extraction of true positives and maybe a few false positives as negatives with  
  `python3 06_improve_data_extraction.py`
4. Choose your newly created path
5. Look at the current image. If it has found the one correct contour or you want to ignore the current image, continue using space.  
  If it has found a false positive or not found the positive contour at all, you can press "n" and collect the false negatives or the missing positives like described in "Start the complete process anew: section 9"
6. You are asked how many positive images you want to create out of the newly extracted positives. Choose a rough number.
7. You are asked how many negative images you want to create out of the newly extracted negatives. Choose a rough number.
8. The augmentation adds the new images in an augmented way to the dataset, this dataset is used for training and evaluation. The output is shown now and can be closed with "q".

### Improve neural network by automatically extracting negatives

The default path for this is big-negative-images. This is only sufficient for the first run so there might be room for adjustments.

1. Activate virtualenv
2. Start the extraction of false positives:  
  `python3 07_extract_false_positives.py`
3. Wait
4. Type your roughly whished size trough augmentation
5. The augmentation adds the new images in an augmented way to the dataset, this dataset is used for training and evaluation. The output is shown now and can be closed with "q".

## Content of cam-config/cam_conf.json
```json
{  
  "cam_target" : 2,  
  "cam_height" : 480,  
  "cam_width" : 640,  
  "time_till_update" : 1,  
  "y_location_text" : 470  
}
```

### cam_target

The "cam_target" describes the camera number. The first should be at 0. If you do not know the camera number, you can search online or just try it out.

### cam_height

The wanted height of a camera frame in pixel.

### cam_width

The wanted width of a camera frame in pixel.

### time_till_update

The detection of an arrow is slowed down to only a few frames to actual read the values. This value describes the pause between the detections in seconds.

### y_location_text

The text does not have a fixed location in the image yet. The default value 470 sets it to the bottom of the image.

References:  
(1) https://docs.opencv.org/4.x/da/d0d/tutorial_camera_calibration_pattern.html  
(2) https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html  
(3) https://stackoverflow.com/questions/17665912/findchessboardcorners-fails-for-calibration-image  
(4) https://github.com/RaubCamaioni/OpenCV_Position (used code)  

