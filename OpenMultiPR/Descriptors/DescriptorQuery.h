#pragma once
#include <vector>
#include "../FileInterface/PicGnssFile.h"
#include "CS/CSOperation.h"
#include "GIST/include/gist.h"
#include "LDB\ldb.h"
#include "BoW\BoW.h"
#include "../FileInterface/GlobalConfig.h"
#include "DescriptorExtraction.h"

class DescriptorQuery
{
public:
	DescriptorQuery(GlobalConfig& config);
	virtual ~DescriptorQuery() ;

	std::vector<std::string> pathsTest;
	std::vector<std::vector<float>> getGISTQuery ()const { return xGISTQuery; };
	cv::Mat getLDBQuery()const { return xLDBQuery; };
	cv::Mat getORBQuery()const { return xORBQuery; };
	cv::Mat getCNNQuery()const { return xCNNQuery; };
	std::vector<std::pair<double, double> >getGPSQuery()const { return xGPSQuery; };

	int getVolume() { return picsTest->getFileVolume(); };
	
	PicGnssFile* picsTest;
private:

	bool isColor;

	/*GPS*/
	std::vector<std::pair<double, double>> xGPSQuery;
	/*GIST*/
	std::vector<std::vector<float>> xGISTQuery;
	/*CNN*/
	cv::Mat xCNNQuery;
	/*LDB*/
	cv::Mat xLDBQuery;
	/*BoW*/
	cv::Mat xORBQuery;

	std::vector<cv::Mat> getAllImage(const cv::Size& imgSize);
	bool extractDescriptor(GlobalConfig& config, arma::Col<klab::DoubleReal>& xCS, std::vector<float>& xGIST, cv::Mat& xLDB, cv::Mat& xORB, cv::Mat& descriptorsSURF);
	void loadFeatureFromFile(const std::string &filename, std::vector<unsigned char>& dim);

	Extraction extraction;
};