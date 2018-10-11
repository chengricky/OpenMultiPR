#include "FileInterface\picGnssfile.h"
#include "descriptors/CS/CSOperation.h"
#include "VisualLocalization.h"
#include "FileInterface\GlobalConfig.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <opencv2/opencv.hpp>

struct Params
{
	int kLDB;
	int kGIST;
	int kBoW;
	double r;
	double numKeyRatio;
	double numKeyTh;
	double PandR[4];
};

GlobalConfig GlobalConfig::config("Config.yaml");

int main()
{
	static GlobalConfig& config = GlobalConfig::instance();
	VisualLocalization vl(config);

	vl.getDistanceMatrix();
	//vl.getEnhancedDistanceMatrix(3);
	vl.showDistanceMatrix();


	cv::waitKey(0);
	return 0;
}
