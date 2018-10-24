#include "FileInterface\picGnssfile.h"
#include "descriptors/CS/CSOperation.h"
#include "VisualLocalization.h"
#include "FileInterface\GlobalConfig.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <opencv2/opencv.hpp>



GlobalConfig GlobalConfig::config("Config.yaml");

int main()
{
	static GlobalConfig& config = GlobalConfig::instance();
	VisualLocalization vl(config);


	vl.getBestMatch();
	//vl.showDistanceMatrix();


	cv::waitKey(0);
	return 0;
}
