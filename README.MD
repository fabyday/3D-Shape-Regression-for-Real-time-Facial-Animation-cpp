# 3D-Shape-Regression-for-Real-time-Facial-Animation c++ predictor
This project focuses solely on prediction code, designed to enhance performance and mitigate Python's inherent execution speed limitations.

Use a Python script to train the regressor.

link to 
[train for regressor](https://github.com/fabyday/3D-Shape-Regression-for-Real-time-Facial-Animation)



# Dependencies
- kinectSDKv2
- OpenCV < 4
- Dlib
- libigl(already included)




# dataset
dlib dataset download it and place it to data/dlib-data
download from [dlib landmark](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)



# Issue Tracker
## eigen library duplication problem issue 1
I use 2-kind of eigen libraries, 
1. opencv with eigen3
2. embeded eigen3 in libigl
these environments make error occuring.(in #include <opencv/core/eigen.hpp>, template error and syntax error will be occured.)
In this reason, I took a shortcut to solve the problem.
the solution is making helper function wrapping them.
