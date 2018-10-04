#pragma once

#include "../Header.h"
#include "LDB\ldb.h"
#include "CS/CSOperation.h"
#include "GIST\include\gist.h"

// command pattern - eliminating decoupling
class ImgDescriptorExtractor
{
protected:
	int imgIdx;//输入rgb-(ir)-(d)图像组，选择处理的图像类别
public:
	ImgDescriptorExtractor(int imgIdx) : imgIdx(imgIdx) {};
	virtual bool extract(std::vector<cv::Mat>  img) = 0;
};

//class SURFExtractor : public ImgDescriptorExtractor
//{	
//	cv::Ptr<cv::FeatureDetector> surfdetector;
//
//	cv::Mat result;
//	vector<cv::KeyPoint> keypoints;
//public:
//	SURFExtractor(cv::Mat const&  img);
//	bool extract();
//	cv::Mat getResult() { return result; };
//};

class ORBExtractor : public ImgDescriptorExtractor
{
	cv::Ptr<cv::ORB> orb;
	cv::Mat result;
	std::vector<cv::KeyPoint> keypoints;

public:
	ORBExtractor(int imgIdx);
	bool extract(std::vector<cv::Mat>   img);
	cv::Mat getResult() { return result; };
	cv::Ptr<cv::ORB> getORB() { return orb; };
	std::vector<cv::KeyPoint> getKeypoints() { return keypoints; };
};

class GISTExtractor : public ImgDescriptorExtractor
{
	bool isNormalize;
	// GIST维度计算=sum(orients)*blocks*blocks*nPics
	cls::GISTParams GIST_PARAMS;
	std::vector<float> result;
public:
	GISTExtractor(int imgIdx, bool useColor, bool isNormalize, cv::Size imgSize ) ;
	bool extract(std::vector<cv::Mat> todoImages);
	std::vector<float> getResult() { return result; };
};

class CSExtractor : public ImgDescriptorExtractor
{
	cv::Size imgSize;
	arma::Col<klab::DoubleReal> result;
public:
	CSExtractor(int imgIdx, cv::Size imgSize);
	bool extract(std::vector<cv::Mat> todoImages);
};

class LDBExtractor : public ImgDescriptorExtractor
{
	bool useColor;
	cv::Mat result;
	cv::Mat illumination_conversion(cv::Mat image);
public:
	LDBExtractor(int imgIdx, bool useColor);
	bool extract(std::vector<cv::Mat> todoImages);
	cv::Mat getResult() { return result; };
};

class Extraction
{
	std::vector<ImgDescriptorExtractor*> extractions;
public:
	virtual ~Extraction();
	void add(ImgDescriptorExtractor* c) { extractions.push_back(c); }
	void run(std::vector<cv::Mat> const&  todoImages);
	ImgDescriptorExtractor* getResult(int idx);
	int getSize() { return extractions.size(); };
};