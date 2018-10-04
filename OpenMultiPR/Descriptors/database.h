#pragma once
#include <vector>
#include "..\FileInterface\PicGnssFile.h"
#include "CS/CSOperation.h"
#include "GIST\include\gist.h"
#include "LDB\ldb.h"
#include "BoW\BoW.h"
#include "..\FileInterface\GlobalConfig.h"
#include "DescriptorExtraction.h"

class Descriptorbase
{
public:	
	Descriptorbase(GlobalConfig& config);
	virtual ~Descriptorbase() {};

	PicGnssFile picsRec;
	std::vector<arma::Col<klab::DoubleReal>> xCSRec;
	std::vector<std::vector<float>> xGISTRec;
	std::vector<cv::Mat> xORBRec;
	cv::Mat xCNNRec;
	cv::Mat xLDBRec;
	cv::Mat xSURFRec;
	DBoW3::Database ORBdb;
	std::vector<std::pair<double, double>> xGPSRec;
	std::vector<int> distanceBase;// 用作距离计算的标签（由于关键点的存在，标签存在问题）1-based
	std::vector<bool> isLabel; //是否是标记点

	int getVolume() { return picsRec.getFileVolume(); };

private:
	std::vector<cv::Mat> getAllImage(const PicGnssFile& picsRec, const cv::Size& imgSize);

	bool isColor;
	Extraction extraction;
};

