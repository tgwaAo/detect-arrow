# distance_measurement_camera
Measure distance to an arrow with your camera

THIS PROJECT IS JUST A PROTOTYPE AND FAR FROM COMPLETE. PLEASE BE CAREFUL AND DO NOT BLINDLY TRUST THIS PROGRAM.

This project was inspired by https://www.pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/.

I thought it would be funny to have an option to measure distance with a camera, because normally you can only track an angle of
what you see, but with this global measurement your robot might get an idea where it could be.

Files:

The "arrow_print.jpg" is the file to be printed. The arrow is the object, that get searched.
"sign_features.npy" is the file of the features to be trained and "sign_target.npy" is the file of targets.
The file named "sign_ai.pkl" contains a trained gaussian progress classifier, so you do not have to train it for yourself.
In "cam_data.txt" the length between the "start" (not the peak) and the center of the arrow and the focal length is stored, so you
should adjust the second value. Both values are multiplicated with each other, so the order is not important.
The file "shape_reference.npy" contains a shape to compare it with found shapes.
"gaussian_process_train.py" loads features and targets and saves ai as "sign_ai.pkl".
"support_functions.py" is as it says, a collection of functions.
"test_real_picture.py" is a file for testing with a local file.
"test_camera.py" is a file to test searching and measurement with your camera.

USAGE:

You can simply start testing with 

python3 test_real_picture.py -i 50_cm.jpg

If you want to build the ai yourself, you can just start "gaussian_process_train.py".

To use your own camera, change the second value of "cam_data.txt" to the focal length of your camera and start "test_camera.py"



