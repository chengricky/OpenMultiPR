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
	std::vector<int> distanceBase;// �����������ı�ǩ�����ڹؼ���Ĵ��ڣ���ǩ�������⣩1-based
	std::vector<bool> isLabel; //�Ƿ��Ǳ�ǵ�

	int getVolume() { return picsRec.getFileVolume(); };

private:
	std::vector<cv::Mat> getAllImage(const PicGnssFile& picsRec, const cv::Size& imgSize);

	bool isColor;
	Extraction extraction;
};

