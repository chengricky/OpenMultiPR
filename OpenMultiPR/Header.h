#pragma once

//#define USE_CONTRIB

#include <opencv2/opencv.hpp>

#ifdef _DEBUG  
#pragma comment(lib, "opencv_world401d.lib")
//#pragma comment(lib, "opencv_core400d.lib")
//#pragma comment(lib, "opencv_highgui400d.lib")
//#pragma comment(lib, "opencv_imgproc400d.lib")
//#pragma comment(lib, "opencv_imgcodecs400d.lib")
//#pragma comment(lib, "opencv_features2d400d.lib")
//#pragma comment(lib, "opencv_dnn400d.lib")
//#pragma comment(lib, "opencv_videoio400d.lib")

#else
#pragma comment(lib, "opencv_world401.lib")
//#pragma comment(lib, "opencv_core400.lib")
//#pragma comment(lib, "opencv_highgui400.lib")
//#pragma comment(lib, "opencv_imgproc400.lib")
//#pragma comment(lib, "opencv_imgcodecs400.lib")
//#pragma comment(lib, "opencv_features2d400.lib")
//#pragma comment(lib, "opencv_dnn400.lib")
//#pragma comment(lib, "opencv_videoio400.lib")

#endif


#if ((defined USE_CONTRIB) && (defined _DEBUG))  
#include <opencv2/xfeatures2d.hpp>
#pragma comment(lib, "opencv_xfeatures2d400d.lib")
#endif  

#if ((defined USE_CONTRIB) && (!defined _DEBUG))  
#include <opencv2/xfeatures2d.hpp>
#pragma comment(lib, "opencv_xfeatures2d400.lib")
#endif 
