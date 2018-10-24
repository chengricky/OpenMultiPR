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
	
private:
	//bool useDepth, useIR, useRGBDIR, useRGBIR; //Ӧ���ǲ��õģ��Ѿ���DesciptorGroup�ж�ȡ��

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

	int chooseGPSDistance(double rangeTh, std::vector<std::pair<double, int>>& xDistance);


	// get a distance matrix, which is as follows
	cv::Mat GISTDistance_RGB, GISTDistance_D, GISTDistance_IR;
	cv::Mat LDBDistance_RGB, LDBDistance_D, LDBDistance_IR;
	cv::Mat GPSDistance;
	cv::Mat GPSMask, GPSMask_uchar;
	cv::Mat CSDistance;
	std::string codeBook;
	//   ----> database
	//  |
	//  |
	//  V
	//  query images
	cv::Mat enhanceMatrix(const cv::Mat& distanceMat, int winSize);

	// ���ƥ����
	std::vector<int> BoWGlobalBest_RGB, BoWGlobalBest_D, BoWGlobalBest_IR;
	std::vector<int> GISTGlobalBest_RGB, GISTGlobalBest_D, GISTGlobalBest_IR;
	std::vector<int> LDBGlobalBest_RGB, LDBGlobalBest_D, LDBGlobalBest_IR;


	GroundTruth ground;
};

// ��pair��������ıȽϺ���
bool cmp(const std::pair<double, int>& a, const std::pair<double, int>& b);
bool cmpPairNum(const std::pair<double, int>& a, const std::pair<double, int>& b);
bool cmpSec(const std::pair<int, int>& a, const std::pair<int, int>& b);

// ����������������ĺ�������
int hamming_matching(cv::Mat desc1, cv::Mat desc2);