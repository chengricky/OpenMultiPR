#include "VisualLocalization.h"
#include "Tools\Timer.h"
#include <direct.h>
#include <cmath>
#include "Tools/GNSSDistance.h"
#include "SequenceSearch.h"
#include "ParameterTuning.h"
#include "FileInterface/GroundTruth.h"

using namespace DBoW3;
using namespace std;

VisualLocalization::VisualLocalization(GlobalConfig& config)
{ 	
	if (!config.getValid())
	{
		throw invalid_argument("Configuration is invalid!");
	}
	descriptorbase = new Descriptors(config, true);
	std::cout << "Database Images is read." << std::endl;
	descriptorquery = new Descriptors(config, false);
	std::cout << "Query Images is read." << std::endl;
	this->codeBook = config.codeBook;
	cv::destroyAllWindows();
	ground.init(config.pathTest + "of.txt", config.pathRec + "of.txt");
	ground.generateGroundTruth(7);
};

VisualLocalization::~VisualLocalization()
{
	if (descriptorbase != nullptr)
	{
		delete descriptorbase; 
	}
	if (descriptorquery !=nullptr)
	{
		delete descriptorquery;
	}
};

bool VisualLocalization::getDistanceMatrix(int channelIdx)
{
	// ���ڲ��Լ����ݺ�ѵ��������,��ȡ��ͬ�����Matrix,MatrixΪempty�����û�и������
	int matRow = descriptorquery->getVolume();
	int matCol = descriptorbase->getVolume();
	// GIST
	cv::Mat GISTQuery, GISTRef, *GISTDistance = nullptr;
	switch (channelIdx)
	{
	case 0: GISTQuery = descriptorquery->GIST_RGB; GISTRef = descriptorbase->GIST_RGB; GISTDistance = &GISTDistance_RGB; break;
	case 1: GISTQuery = descriptorquery->GIST_D; GISTRef = descriptorbase->GIST_D; GISTDistance = &GISTDistance_D; break;
	case 2: GISTQuery = descriptorquery->GIST_IR; GISTRef = descriptorbase->GIST_IR; GISTDistance = &GISTDistance_IR; break;
	default:
		break;
	}
	if (GISTQuery.empty() || GISTRef.empty())
	{
		*GISTDistance = cv::Mat();
	}
	else
	{
		*GISTDistance = cv::Mat(matRow, matCol, CV_32FC1);
		for (size_t i = 0; i < matRow; i++)
			for (size_t j = 0; j < matCol; j++)
				GISTDistance->at<float>(i, j) = cv::norm(GISTQuery.row(i), GISTRef.row(j), cv::NORM_L2);
	}
	// LDB
	cv::Mat LDBQuery, LDBRef, *LDBDistance = nullptr;
	switch (channelIdx)
	{
	case 0: LDBQuery = descriptorquery->LDB_RGB; LDBRef = descriptorbase->LDB_RGB; LDBDistance = &LDBDistance_RGB; break;
	case 1: LDBQuery = descriptorquery->LDB_D; LDBRef = descriptorbase->LDB_D; LDBDistance = &LDBDistance_D; break;
	case 2: LDBQuery = descriptorquery->LDB_IR; LDBRef = descriptorbase->LDB_IR; LDBDistance = &LDBDistance_IR; break;
	default:
		break;
	}
	if (LDBQuery.empty() || LDBRef.empty())
	{
		*LDBDistance = cv::Mat();
	}
	else
	{
		*LDBDistance = cv::Mat(matRow, matCol, CV_32FC1);
		for (size_t i = 0; i < matRow; i++)
			for (size_t j = 0; j < matCol; j++)
				LDBDistance->at<float>(i, j) = hamming_matching(LDBQuery.row(i), LDBRef.row(j));
	}
	return true;
}

bool VisualLocalization::getDistanceMatrix(float gnssTh)
{
	int matRow = descriptorquery->getVolume();
	int matCol = descriptorbase->getVolume();
	// CS
	auto CSQuery = descriptorquery->CS;
	auto CSRef = descriptorbase->CS;
	if (CSQuery.empty() || CSRef.empty())
	{
		CSDistance = cv::Mat();
	}
	else
	{
		CSDistance = cv::Mat(matRow, matCol, CV_32FC1);
		for (size_t i = 0; i < matRow; i++)
			for (size_t j = 0; j < matCol; j++)
				CSDistance.at<float>(i, j) = cv::norm(CSQuery.row(i), CSRef.row(j), cv::NORM_L2);
	}
	// GPS
	auto GPSQuery = descriptorquery->GPS;
	auto GPSRef = descriptorbase->GPS;
	if (GPSQuery.empty() || GPSRef.empty())
	{
		GPSDistance = cv::Mat();
		GPSMask = cv::Mat();
	}
	else
	{
		GPSDistance = cv::Mat(matRow, matCol, CV_32FC1);
		for (size_t i = 0; i < matRow; i++)
			for (size_t j = 0; j < matCol; j++)
				GPSDistance.at<float>(i, j) = GNSSdistance(GPSQuery.at<float>(i,1), GPSQuery.at<float>(i, 0), GPSRef.at<float>(j, 1), GPSRef.at<float>(j, 0));
		cv::threshold(GPSDistance, GPSMask, gnssTh, FLT_MAX, cv::THRESH_BINARY_INV);
		GPSMask.convertTo(GPSMask_uchar, CV_8UC1);
	}
	
	return true;
}

bool VisualLocalization::getGlobalSearch(int channelIdx)
{
	// BoW
	std::vector<cv::Mat> ORBQuery, ORBRef;
	std::vector<int> *BoWGlobalBest = nullptr;
	switch (channelIdx)
	{
	case 0: ORBQuery = descriptorquery->ORB_RGB; ORBRef = descriptorbase->ORB_RGB; BoWGlobalBest = &BoWGlobalBest_RGB; break;
	case 1: ORBQuery = descriptorquery->ORB_D; ORBRef = descriptorbase->ORB_D; BoWGlobalBest = &BoWGlobalBest_D;  break;
	case 2: ORBQuery = descriptorquery->ORB_IR; ORBRef = descriptorbase->ORB_IR; BoWGlobalBest = &BoWGlobalBest_IR; break;
	default:
		break;
	}
	if (ORBQuery.empty() || ORBRef.empty())
	{
		(*BoWGlobalBest) = cv::Mat();
	}
	else
	{
#ifdef _DEBUG
		(*BoWGlobalBest) = cv::Mat();
#else
		// load the vocabulary from disk
		DBoW3::Vocabulary voc(codeBook);
		DBoW3::Database ORBdb;
		ORBdb.setVocabulary(voc, false, 0); // false = do not use direct index (so ignore the last param)
											 //The direct index is useful if we want to retrieve the features that belong to some vocabulary node.
											 //db creates a copy of the vocabulary, we may get rid of "voc" now, add images to the database
											 //loop for every images of training dataset
		for (size_t i = 0; i < ORBRef.size(); i++)
		{
			ORBdb.add(ORBRef[i]);
		}
		DBoW3::QueryResults ret;
		for (size_t i = 0; i < ORBQuery.size(); i++)
		{
			
			if (GPSMask.empty())
			{
				ORBdb.query(ORBQuery[i], ret);
				BoWGlobalBest->push_back(ret[0].Id);
			}
			else
			{
				ORBdb.query(ORBQuery[i], ret, -1);//ret ��0-based
				for (auto r : ret)
				{
					if (GPSMask.at<uchar>(i, r.Id))
					{
						BoWGlobalBest->push_back(r.Id);
						break;
					}
				}
			}
		}
#endif // DEBUG
	}

	cv::Mat LDBdistanceMat, GISTdistanceMat;
	std::vector<int> *LDBglobalResult = nullptr, *GISTglobalResult = nullptr;
	switch (channelIdx)
	{
	case 0: LDBdistanceMat = LDBDistance_RGB; GISTdistanceMat = GISTDistance_RGB; LDBglobalResult = &LDBGlobalBest_RGB; GISTglobalResult = &GISTGlobalBest_RGB; break;
	case 1: LDBdistanceMat = LDBDistance_D; GISTdistanceMat = GISTDistance_D; LDBglobalResult = &LDBGlobalBest_D; GISTglobalResult = &GISTGlobalBest_D; break;
	case 2: LDBdistanceMat = LDBDistance_IR; GISTdistanceMat = GISTDistance_IR; LDBglobalResult = &LDBGlobalBest_IR; GISTglobalResult = &GISTGlobalBest_IR;  break;
	default:
		break;
	}
	LDBdistanceMat = LDBdistanceMat & GPSMask;
	GISTdistanceMat = GISTdistanceMat & GPSMask;
	LDBdistanceMat.setTo(FLT_MAX, ~GPSMask_uchar);
	GISTdistanceMat.setTo(FLT_MAX, ~GPSMask_uchar);

	for (size_t i = 0; i < LDBdistanceMat.rows; i++)//query
	{
		//When minIdx is not NULL, it must have at least 2 elements (as well as maxIdx), even if src is a single-row or single-column matrix.
		//In OpenCV (following MATLAB) each array has at least 2 dimensions, i.e. single-column matrix is Mx1 matrix (and therefore minIdx/maxIdx will be (i1,0)/(i2,0)) 
		//and single-row matrix is 1xN matrix (and therefore minIdx/maxIdx will be (0,j1)/(0,j2)).
		int* minPos = new int[2];
		cv::minMaxIdx(LDBdistanceMat.row(i), nullptr, nullptr, minPos, nullptr);	//�ɷ�ĳ�top-k		?
		LDBglobalResult->push_back(minPos[1]);
		cv::minMaxIdx(GISTdistanceMat.row(i), nullptr, nullptr, minPos, nullptr);
		GISTglobalResult->push_back(minPos[1]);
		delete minPos;
	}
	return true;
}


void VisualLocalization::getBestMatch()
{
	// generate distance matric for gist \ ldb \ cs and gps && BoW 
	// get global best idx
	getDistanceMatrix((float)15);
	for (size_t i = 0; i < 3; i++)
	{
		getDistanceMatrix((int)i);
		getGlobalSearch(i);
	}
	
	int matRow = descriptorquery->getVolume();
	int matCol = descriptorbase->getVolume();
	cv::Size matSize(matCol, matRow);

	Parameter2F1 pt(ground.gt, BoWGlobalBest_RGB, BoWGlobalBest_D, BoWGlobalBest_IR, GISTGlobalBest_RGB, GISTGlobalBest_D, GISTGlobalBest_IR,
		LDBGlobalBest_RGB, LDBGlobalBest_D, LDBGlobalBest_IR, matSize);

	//// use OpenGA to optimize coefficents
	std::vector<double> coeff;
	pt.prepare4MultimodalCoefficients();	
	optimizeMultimodalCoefficients(&pt, coeff);
	pt.updateParams(coeff);

	////// calculate score matrix for single descriptor
	pt.placeRecognition();
	//pt.printMatchingResults();
	std::cout << pt.calculateF1score() << std::endl;

	// sweep the parameter
	//std::vector<double> coeff = { 0.49, 0.03, 0.45, 0.30, 0.02, 0.12, 0.67, 0.87, 0.55 };
	//pt.updateParams(coeff);
	//for (float i = 0; i < 0.2; i+=0.02)
	//{
	//	pt.updateParams(i);
	//	pt.placeRecognition();
	//	std::cout << pt.calculateF1score() << std::endl;
	//}


}

static bool getDistanceMap(cv::Mat DistanceMat, std::string DescriptorTtype)
{
	cv::Mat CS_norm;
	cv::normalize(DistanceMat, CS_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	cv::resize(CS_norm, CS_norm, cv::Size(CS_norm.cols * 5, CS_norm.rows * 5));
	cv::imshow("Distance Matrix - "+ DescriptorTtype, CS_norm);
	return true;
}

bool VisualLocalization::showDistanceMatrix()
{
	if (!CSDistance.empty())
	{
		getDistanceMap(CSDistance, "CS");
	}
	if (!LDBDistance_RGB.empty())
	{
		getDistanceMap(LDBDistance_RGB, "LDB");
	}
	if (!LDBDistance_D.empty())
	{
		getDistanceMap(LDBDistance_D, "LDB");
	}
	if (!LDBDistance_IR.empty())
	{
		getDistanceMap(LDBDistance_IR, "LDB");
	}
	if (!GPSDistance.empty())
	{
		getDistanceMap(GPSDistance, "GPS");
	}
	if (!GISTDistance_RGB.empty())
	{
		getDistanceMap(GISTDistance_RGB, "GIST");
	}
	if (!GISTDistance_D.empty())
	{
		getDistanceMap(GISTDistance_D, "GIST");
	}
	if (!GISTDistance_IR.empty())
	{
		getDistanceMap(GISTDistance_IR, "GIST");
	}
	cv::waitKey(1);
	return true;
}

cv::Mat VisualLocalization::enhanceMatrix(const cv::Mat& distanceMat, int winSize) //winSize����Ϊ����
{
	assert(winSize % 2);
	cv::Mat enhancedMat(distanceMat.size(), CV_32FC1);
	if (!distanceMat.empty())
	{
		for (size_t i = 0; i < distanceMat.rows; i++)
		{
			const float* pDis = distanceMat.ptr<float>(i);
			float* pEnh = enhancedMat.ptr<float>(i);
			for (size_t j = 0; j < (winSize-1)/2; j++)
			{
				float sum = std::accumulate(pDis, pDis + winSize, 0.0f);//�Ƿ���У�
				float mean = sum / winSize;
				float accum = 0.0f;
				std::for_each(pDis, pDis + winSize, [&](const float d) {
					accum += (d - mean)*(d - mean);
				});
				float stdev = sqrt(accum / (winSize - 1)); //��׼��
				pEnh[j] = (pDis[j]-mean) / stdev;
			}
			for (size_t j = 0; j < distanceMat.cols-(winSize-1); j++)
			{
				float sum = std::accumulate(pDis+j, pDis+j + winSize, 0.0f);//�Ƿ���У�
				float mean = sum / winSize;
				float accum = 0.0f;
				std::for_each(pDis + j, pDis + j + winSize, [&](const float d) {
					accum += (d - mean)*(d - mean);
				});
				float stdev = sqrt(accum / (winSize - 1)); //��׼��
				pEnh[j+ (winSize - 1) / 2] = (pDis[j + (winSize - 1) / 2] - mean) / stdev;
			}
			for (size_t j = distanceMat.cols - (winSize - 1) / 2; j < distanceMat.cols; j++)
			{
				float sum = std::accumulate(pDis + distanceMat.cols - winSize, pDis + distanceMat.cols, 0.0f);//�Ƿ���У�
				float mean = sum / winSize;
				float accum = 0.0f;
				std::for_each(pDis, pDis + winSize, [&](const float d) {
					accum += (d - mean)*(d - mean);
				});
				float stdev = sqrt(accum / (winSize - 1)); //��׼��
				pEnh[j] = (pDis[j] - mean) / stdev;
			}
		}
	}
	else
	{
		enhancedMat = distanceMat;
	}
	return enhancedMat;
}

bool VisualLocalization::getEnhancedDistanceMatrix(int winSize)
{
	//CSDistance = enhanceMatrix(CSDistance, winSize);
	//GISTDistance = enhanceMatrix(GISTDistance, winSize);
	//GPSDistance = enhanceMatrix(GPSDistance, winSize);
	//LDBDistance = enhanceMatrix(LDBDistance, winSize);
	return true;
}

int VisualLocalization::chooseGPSDistance(double rangeTh, std::vector<std::pair<double, int>>& xGPSDistance)
{
	// �����滻�ɶ����򣬿ɽ���ʱ�临�Ӷȵ�nLogK?
	std::sort(xGPSDistance.begin(), xGPSDistance.end(), cmp);
	int GPSrear = 0;
	for (size_t i = 0; i < xGPSDistance.size(); i++)
	{
		if (xGPSDistance[i].first > rangeTh)
		{
			GPSrear = i;
			break;
		}
	}
	return max(GPSrear,30);
	//return GPSrear;
}

bool cmp(const std::pair<double, int>& a, const std::pair<double, int>& b)
{
	return a.first < b.first;
}

bool cmpPairNum(const std::pair<double, int>& a, const std::pair<double, int>& b)
{
	return a.first > b.first;
}

bool cmpSec(const std::pair<int, int>& a, const std::pair<int, int>& b)
{
	return a.second < b.second;
}

/**
* @brief This method computes the Hamming distance between two binary
* descriptors
* @param desc1 First descriptor
* @param desc2 Second descriptor
* @return Hamming distance between the two descriptors
*/
int hamming_matching(cv::Mat desc1, cv::Mat desc2) {

	int distance = 0;

	if (desc1.rows != desc2.rows || desc1.cols != desc2.cols || desc1.rows != 1 || desc2.rows != 1) {

		std::cout << "The dimension of the descriptors is different." << std::endl;
		return -1;

	}

	for (int i = 0; i < desc1.cols; i++) {
		distance += (*(desc1.ptr<unsigned char>(0) + i)) ^ (*(desc2.ptr<unsigned char>(0) + i));
	}

	return distance;

}

