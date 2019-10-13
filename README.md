# distance_measurement_camera
Measure distance to an arrow with your camera.

THIS PROJECT IS JUST A PROTOTYPE AND FAR FROM COMPLETE. PLEASE BE CAREFUL WHEN YOU USE IT IN YOUR PROJECTS.

This project was inspired by https://www.pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/.

I thought it would be funny to have an option to measure distance with a camera, because normally you can only track an angle of
what you see, but with this global measurement your robot might get an idea where it could be and in which direction it should drive.

Files:

The "arrow_print.jpg" is the file to be printed. The arrow is the object, that get searched.
"sign_features.npy" is the file of the features to be trained and "sign_target.npy" is the file of targets.
The file named "sign_ai.pkl" contains a trained gaussian progress classifier, so you do not have to train it for yourself.
In "cam_data.txt" the length between the "start" (not the peak) and the center of the arrow is the first number and the second is the focal length in pixels, so you
should adjust the second value for EVERY RESOLUTION. Both values are multiplicated with each other, so the order is not important.
The file "shape_reference.npy" contains a shape to compare it with found shapes.
"gaussian_process_train.py" loads features and targets and saves ai as "sign_ai.pkl".
"support_functions.py" is as it says, a collection of functions.
"test_real_picture.py" is a file for testing with a local file.
"test_camera.py" is a file to test searching and measurement with your camera.

USAGE:

Adjust the second parameter in "cam_data.txt" to your camera. The first is 126.5 and should not be changed. The second has a value of 730 for my example picture and the value for 1280 * 960 is around 1117. Please recognize that your paper could be rotated, which results in a different distance.

You can simply start testing with 

python3 test_real_picture.py -i 50_cm.jpg

If you want to build the ai yourself, you can just start "gaussian_process_train.py".

To use your own camera, change the second value of "cam_data.txt" to the focal length of your camera in pixels and start "test_camera.py"



