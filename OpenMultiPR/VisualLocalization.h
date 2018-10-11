#pragma once

#include "Descriptors\descriptorgroup.h"
#include "Descriptors/BoW/BoW.h"

class VisualLocalization
{
public:
	VisualLocalization(GlobalConfig& config);
	virtual ~VisualLocalization();

	bool showDistanceMatrix();
	bool getDistanceMatrix();
	bool getEnhancedDistanceMatrix(int winSize);

	
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
	cv::Mat GISTDistance;
	cv::Mat LDBDistance;
	cv::Mat GPSDistance;
	cv::Mat CSDistance;
	cv::Mat BoWDistance;
	std::string codeBook;
	//   ----> database
	//  |
	//  |
	//  V
	//  query images
	cv::Mat enhanceMatrix(const cv::Mat& distanceMat, int winSize);


};

// ��pair��������ıȽϺ���
bool cmp(const std::pair<double, int>& a, const std::pair<double, int>& b);
bool cmpPairNum(const std::pair<double, int>& a, const std::pair<double, int>& b);
bool cmpSec(const std::pair<int, int>& a, const std::pair<int, int>& b);

// ����������������ĺ�������
int hamming_matching(cv::Mat desc1, cv::Mat desc2);