# detect_arrow

THIS PROJECT IS JUST A PROTOTYPE AND FAR FROM COMPLETE. THERE IS NO WARANTY FOR A FUNCTION WITHOUT FAILURES.

The project allows the user to detect the given arrow, which can be printed and sticked on a wall. The arrow is detected via its contours and additional information like the position in the image, the distance to the camera and pointing direction can are calculated.

The distance is measured as described in https://www.pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/.

USAGE:

You can simply start testing with 

python3 test_picture.py -i ../50_cm.jpg

To use your own pictures, adjust the second parameter in "cam_data.txt" to your camera by the formular

first_value_of_cam_data * second_value_of_cam_data / (size_in_pixels * 10)

like in the link above. Please recognize that your paper could be rotated, which results in a different distance. The first value in cam_data.txt is 126.5 and should not be changed. The number in the example is 760. 

If you want to build the ai yourself, you can just start "train_arrows.py".

Files:

The "arrow_print.jpg" is the file to be printed. The arrow is the object, that get searched.
"sign_features.npy" is the file of the features to be trained and "sign_target.npy" is the file of targets.
The file named "sign_ai.pkl" contains a trained gaussian progress classifier, so you do not have to train it for yourself.
In "cam_data.txt" the length between the "start" (not the peak) and the center of the arrow is the first number and the second is the focal length in pixels, so you
should adjust the second value for EVERY RESOLUTION and lense. Both values are multiplicated with each other, so the order is not important.
The file "shape_reference.npy" contains a shape to compare it with found shapes.
"train_arrows.py" loads features and targets and saves ai as "sign_ai.pkl".
"extract_features.py" extracts features to use them for comparison in ai.
"test_picture.py" is a file for testing with a local file.
"test_camera.py" is a file to test searching and measurement with your camera.

