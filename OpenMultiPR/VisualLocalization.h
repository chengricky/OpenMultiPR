#pragma once
#include "FileInterface\picGnssfile.h"
#include "standalone_image.h"
#include "gist.h"
#include "Descriptors\BoW\BoW.h"
#include "FileInterface\GlobalConfig.h"
#include "Descriptors\database.h"
#include "Descriptors\DescriptorQuery.h"


class VisualLocalization
{
public:
	VisualLocalization(GlobalConfig& config);
	virtual ~VisualLocalization();
	
	// approach to finding good matches, �Ƿ�����������Ҫ�б����Ϣ�����߹ؼ���ľ����ٻ���
	bool test(GlobalConfig & config, const bool& recordIdxDev, const bool &outPR, double* PandR, int kLDB=5, int kGIST=5, int kBoW=5, double r=0.02, double numKeyRatio=0.5, double numKeyTh=3);
	// approach to finding good matches, �Ƿ�����������Ҫ�б����Ϣ�����߹ؼ���ľ����ٻ���
	bool paperLocalize(const bool & recordIdxErr, double* PandR, const bool & outPR = true, int kLDB = 5, int kGIST = 5, int kBoW = 5, double r = 0.02, double numKeyRatio = 0.5, double numKeyTh = 3);
	/// �����GPS��λ
	bool GPSLocalize();
	
private:
	bool useDepth, useIR, useRGBDIR, useRGBIR;

	// ѵ��������(�����¼��·��)
	Descriptorbase* descriptorbase;

	std::vector<std::vector<std::pair<double, int>>> gist;
	std::vector<std::vector<std::pair<double, int>>> ldb;
	std::vector<std::vector<std::pair<double, int>>> gps;
	std::vector<DBoW3::QueryResults> bow;

	// ���Լ�����
	DescriptorQuery* descriptorquery;
	std::vector<bool> keyGT, keyPredict, keyGPS;
	bool withGPS;
	std::string descriptor;


	bool outputRet(std::ofstream & fResult, const std::vector<std::pair<double, int>>& xDistance, const int& bestMatchValue, const std::string& path, const int& picIdx);
	bool outputRet(std::ofstream & fResult, const std::vector<std::pair<double, int>>& xDistance, const int& bestMatchValue, const int& k, const std::string& path, const int& picIdx);
	int chooseGPSDistance(double rangeTh, std::vector<std::pair<double, int>>& xDistance);
	bool getMatchingResult(const std::vector<std::pair<double, int>>& xGPSDistance, int GPSrear, 
		const std::vector<std::pair<double, int>>& xDistance, int k, std::vector<std::pair<double, int>>& ret);
	bool getMatchingResult(const std::vector<std::pair<double, int>>& xGPSDistance, int GPSrear,
		const DBoW3::QueryResults& ret, int k, std::vector<std::pair<double, int>>& retBoW);

	void getfStreamOut(std::string pathTest, std::ofstream* fResult, std::vector<std::string>& pathsOut, int);

	// get a distance matrix, which is as follows
	//   ----> database
	//  |
	//  |
	//  V
	//  query images
	
	/// ����GIST�����γɾ������
	void getGISTDistance(cv::Mat& GISTDistance);
	/// ����LDB�����γɾ������
	void getLDBDistance(cv::Mat& LDBDistance);
	/// ����GPS�����γɾ������
	void getGPSDistance(cv::Mat& GPSDistance);
	/// ����ORB�����γɾ������
	void getORBDistance(cv::Mat& ORBDistance);

	//void getDistance(cv::Mat& GISTDistance, cv::Mat& LDBDistance,
	//	/*std::vector<std::pair<double, int>>& CSDistance,*/
	//	cv::Mat& GPSDistance, cv::Mat& ORBDistance/*, DBoW3::QueryResults& ret*/);

	void getMatchingResultfromDistance(const cv::Mat& in, std::vector<std::pair<double, int>>& outResult, const int& i);

	void getMatchingResultfromPair(const cv::Mat& in, std::vector<std::pair<double, int>>& outResult);


};

// ��pair��������ıȽϺ���
bool cmp(const std::pair<double, int>& a, const std::pair<double, int>& b);
bool cmpPairNum(const std::pair<double, int>& a, const std::pair<double, int>& b);
bool cmpSec(const std::pair<int, int>& a, const std::pair<int, int>& b);

// ����������������ĺ�������
int hamming_matching(cv::Mat desc1, cv::Mat desc2);