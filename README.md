# distance_measurement_camera
Measure distance to an arrow with your camera.

THIS PROJECT IS JUST A PROTOTYPE AND FAR FROM COMPLETE. THERE IS NO WARANTY FOR A FUNCTION WITHOUT FAILURES.

The project allows the user to detect the given arrow. This arrow can be printed and sticked on a wall. In this prototype, the arrow gets detected via its contours and data like the position in the image, the distance to the camera and pointing direction can be used.

The distance is measured as described in https://www.pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/.


Files:

The "arrow_print.jpg" is the file to be printed. The arrow is the object, that get searched.
"sign_features.npy" is the file of the features to be trained and "sign_target.npy" is the file of targets.
The file named "sign_ai.pkl" contains a trained gaussian progress classifier, so you do not have to train it for yourself.
In "cam_data.txt" the length between the "start" (not the peak) and the center of the arrow is the first number and the second is the focal length in pixels, so you
should adjust the second value for EVERY RESOLUTION. Both values are multiplicated with each other, so the order is not important.
The file "shape_reference.npy" contains a shape to compare it with found shapes.
"train_arrows.py" loads features and targets and saves ai as "sign_ai.pkl".
"extract_features.py" extracts features to use them for comparison in ai.
"test_picture.py" is a file for testing with a local file.
"test_camera.py" is a file to test searching and measurement with your camera.

USAGE:

Adjust the second parameter in "cam_data.txt" to your camera. The first is 126.5 and should not be changed. The second has a value of 730 for my example picture and the value for 1280 * 960 is around 1117. Please recognize that your paper could be rotated, which results in a different distance.

You can simply start testing with 

python3 test_picture.py -i 50_cm.jpg

If you want to build the ai yourself, you can just start "train_arrows.py".

To use your own camera, change the second value of "cam_data.txt" to the focal length of your camera in pixels and start "test_camera.py"



