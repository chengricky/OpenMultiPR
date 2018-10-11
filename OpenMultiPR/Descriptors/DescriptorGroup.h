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
	cv::Mat GIST;
	//cv::Mat CNN;
	cv::Mat LDB;
	std::vector<cv::Mat> ORB;
	//std::vector<cv::Mat> SURF;

	cv::Mat GPS;

	int getVolume() { return picFiles.getFileVolume(); };

private:
	std::vector<cv::Mat> getAllImage(const PicGNSSFile& picsRec, const cv::Size& imgSize);

	bool isColor;
	Extraction extraction;
};