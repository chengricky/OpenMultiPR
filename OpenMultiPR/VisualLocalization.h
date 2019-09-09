#pragma once

#include "Descriptors\descriptorgroup.h"
#include "Descriptors/BoW/BoW.h"
#include "FileInterface/GroundTruth.h"

class VisualLocalization
{
public:
	VisualLocalization(GlobalConfig& config);
	virtual ~VisualLocalization();

	void getBestMatch();
	bool showDistanceMatrix();
	bool getDistanceMatrix(int channelIdx);
	bool getDistanceMatrix(float GNSS=30);
	bool getEnhancedDistanceMatrix(int winSize);
	bool getGlobalSearch(int channelIdx);
	bool getGlobalSearch();

	bool generateVideo(std::vector<int> matchingResults, std::string path="");
	
private:
	// ѵ��������(�����¼��·��)
	Descriptors* descriptorbase;

	std::vector<std::vector<std::pair<double, int>>> gist;
	std::vector<std::vector<std::pair<double, int>>> ldb;
	std::vector<std::vector<std::pair<double, int>>> gps;
	std::vector<DBoW3::QueryResults> bow;

	// ���Լ�����
	Descriptors* descriptorquery;
	std::vector<bool> keyGT, keyPredict, keyGPS;
	bool withGPS;
	std::string descriptor;

	/// get a distance matrix, which is as follows:	
	cv::Mat GISTDistance[3];	//GISTDistance_RGB, GISTDistance_D, GISTDistance_IR;
	cv::Mat LDBDistance[3];	//LDBDistance_RGB, LDBDistance_D, LDBDistance_IR;
	cv::Mat GPSDistance;
	cv::Mat GPSMask_uchar;
	cv::Mat CSDistance;
	cv::Mat GGDistance;
	std::string codeBook;
	//   ----> database
	//  |
	//  |
	//  V
	//  query images
	cv::Mat enhanceMatrix(const cv::Mat& distanceMat, int winSize);
	int matRow, matCol;


	// ���ƥ����
	std::vector<int> BoWGlobalBest_RGB, BoWGlobalBest_IR;
	std::vector<int> GISTGlobalBest_RGB, GISTGlobalBest_D, GISTGlobalBest_IR;
	std::vector<int> LDBGlobalBest_RGB, LDBGlobalBest_D, LDBGlobalBest_IR;
	std::vector<int> GPSGlobalBest;
	std::vector<int> GGglobalResult;

	GroundTruth ground;

	int runningMode;
};

// ����������������ĺ�������
int hamming_matching(cv::Mat desc1, cv::Mat desc2);