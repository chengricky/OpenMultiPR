#pragma once
#include <vector>
#include "..\FileInterface\PicGnssFile.h"
#include "CS/CSOperation.h"
#include "GIST\include\gist.h"
#include "LDB\ldb.h"
#include "..\FileInterface\GlobalConfig.h"
#include "DescriptorExtraction.h"

class Descriptors
{
public:
	Descriptors(GlobalConfig& config, bool isRefImage);
	virtual ~Descriptors() {};

	PicGNSSFile picFiles;
	std::string picPath;
	cv::Mat CS;
	// 按顺序分别是RGB\D\IR图像的描述子
	cv::Mat GIST_RGB, GIST_D, GIST_IR;
	cv::Mat LDB_RGB, LDB_D, LDB_IR;
	std::vector<cv::Mat> ORB_RGB, ORB_D, ORB_IR;
	//std::vector<cv::Mat> SURF;

	cv::Mat GPS;

	int getVolume() { return picFiles.getFileVolume(); };

private:
	std::vector<cv::Mat> getAllImage(const PicGNSSFile& picsRec, const cv::Size& imgSize);

	bool isColor;
	Extraction extraction;
};