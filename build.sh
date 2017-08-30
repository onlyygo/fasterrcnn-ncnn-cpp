g++ -fopenmp -std=c++11 Demo_Detection.cpp FasterRCNN.cpp libncnn.a -lopencv_core -lopencv_highgui -lopencv_imgproc \
-I./ncnn_include \
-I/usr/include \
-I/usr/include/opencv \
-I/usr/include/opencv2 \
-o Demo_Detection
