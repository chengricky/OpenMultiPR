#pragma once

#define USE_CONTRIB

#include <opencv2/opencv.hpp>
#ifdef USE_CONTRIB
#include <opencv2/xfeatures2d.hpp>
#endif

#ifdef _DEBUG  
#pragma comment(lib, "opencv_core343d.lib")
#pragma comment(lib, "opencv_highgui343d.lib")
#pragma comment(lib, "opencv_imgproc343d.lib")
#pragma comment(lib, "opencv_imgcodecs343d.lib")
#pragma comment(lib, "opencv_features2d343d.lib")
#pragma comment(lib, "opencv_xfeatures2d343d.lib")
#else
#pragma comment(lib, "opencv_core343.lib")
#pragma comment(lib, "opencv_highgui343.lib")
#pragma comment(lib, "opencv_imgproc343.lib")
#pragma comment(lib, "opencv_imgcodecs343.lib")
#pragma comment(lib, "opencv_features2d343.lib")
#pragma comment(lib, "opencv_xfeatures2d343.lib")
#endif