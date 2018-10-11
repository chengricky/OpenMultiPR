#include "VisualLocalization.h"
#include "Tools\Timer.h"
#include <direct.h>
#include <cmath>

using namespace DBoW3;
using namespace std;

// Get Time Stamp with the style of yyyy-mm-dd_hh-mm-ss
std::string getTimeStamp()
{
	time_t timep;
	timep = time(NULL); /* get current time */
	struct tm *p;
	p = localtime(&timep); /*转换为struct tm结构的当地时间*/
	stringstream timeStampStream;
	timeStampStream << 1900 + p->tm_year << setw(2) << setfill('0') << 1 + p->tm_mon << setw(2) << setfill('0') << p->tm_mday << "_";
	timeStampStream << p->tm_hour << "-" << p->tm_min << "-" << p->tm_sec;
	return timeStampStream.str();
}

bool cmpPair(const std::pair<double, int>& a, const std::pair<double, int>& b)
{
	return(a.first < b.first);
}

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



bool VisualLocalization::getDistanceMatrix()
{
	// 对于测试集数据和训练集数据,获取不同距离的Matrix,Matrix为empty则代表没有该项距离
	int matRow = descriptorquery->getVolume();
	int matCol = descriptorbase->getVolume();
	// GIST
	auto GISTQuery = descriptorquery->GIST;
	auto GISTRef = descriptorbase->GIST;
	if (GISTQuery.empty() || GISTRef.empty())
	{
		GISTDistance = cv::Mat();
	}
	else
	{
		GISTDistance = cv::Mat(matRow, matCol, CV_32FC1);
		for (size_t i = 0; i < matRow; i++)
			for (size_t j = 0; j < matCol; j++)
				GISTDistance.at<float>(i, j) = cv::norm(GISTQuery.row(i), GISTRef.row(j), cv::NORM_L2);
	}
	// LDB
	auto LDBQuery = descriptorquery->LDB;
	auto LDBRef = descriptorbase->LDB;
	if (LDBQuery.empty() || LDBRef.empty())
	{
		LDBDistance = cv::Mat();
	}
	else
	{
		LDBDistance = cv::Mat(matRow, matCol, CV_32FC1);
		for (size_t i = 0; i < matRow; i++)
			for (size_t j = 0; j < matCol; j++)
				LDBDistance.at<float>(i, j) = hamming_matching(LDBQuery.row(i), LDBRef.row(j));
	}
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
	}
	else
	{
		GPSDistance = cv::Mat(matRow, matCol, CV_32FC1);
		for (size_t i = 0; i < matRow; i++)
			for (size_t j = 0; j < matCol; j++)
				GPSDistance.at<float>(i, j) = cv::norm(GPSQuery.row(i), GPSRef.row(j), cv::NORM_L2);
	}
	// BoW
	auto ORBQuery = descriptorquery->ORB;
	std::vector<cv::Mat> ORBRef = descriptorbase->ORB;
	if (ORBQuery.empty()|| ORBRef.empty())
	{
		BoWDistance = cv::Mat();
	}
	else
	{
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
		//BoWDistance = cv::Mat(matRow, matCol, CV_32FC1);
		for (size_t i = 0; i < ORBQuery.size(); i++)
		{
			ORBdb.query(ORBQuery[i], ret, -1);			
			for (size_t j = 0; j < matCol; j++)
				;
				//BoWDistance.at<float>(i, ret[j].Id) = ret[j].Score;
		}
	}
	return true;
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
	if (!LDBDistance.empty())
	{
		getDistanceMap(LDBDistance, "LDB");
	}
	if (!GPSDistance.empty())
	{
		getDistanceMap(GPSDistance, "GPS");
	}
	if (!GISTDistance.empty())
	{
		getDistanceMap(GISTDistance, "GIST");
	}
	cv::waitKey(1);
	return true;
}

cv::Mat VisualLocalization::enhanceMatrix(const cv::Mat& distanceMat, int winSize) //winSize必须为奇数
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
				float sum = std::accumulate(pDis, pDis + winSize, 0.0f);//是否可行？
				float mean = sum / winSize;
				float accum = 0.0f;
				std::for_each(pDis, pDis + winSize, [&](const float d) {
					accum += (d - mean)*(d - mean);
				});
				float stdev = sqrt(accum / (winSize - 1)); //标准差
				pEnh[j] = (pDis[j]-mean) / stdev;
			}
			for (size_t j = 0; j < distanceMat.cols-(winSize-1); j++)
			{
				float sum = std::accumulate(pDis+j, pDis+j + winSize, 0.0f);//是否可行？
				float mean = sum / winSize;
				float accum = 0.0f;
				std::for_each(pDis + j, pDis + j + winSize, [&](const float d) {
					accum += (d - mean)*(d - mean);
				});
				float stdev = sqrt(accum / (winSize - 1)); //标准差
				pEnh[j+ (winSize - 1) / 2] = (pDis[j + (winSize - 1) / 2] - mean) / stdev;
			}
			for (size_t j = distanceMat.cols - (winSize - 1) / 2; j < distanceMat.cols; j++)
			{
				float sum = std::accumulate(pDis + distanceMat.cols - winSize, pDis + distanceMat.cols, 0.0f);//是否可行？
				float mean = sum / winSize;
				float accum = 0.0f;
				std::for_each(pDis, pDis + winSize, [&](const float d) {
					accum += (d - mean)*(d - mean);
				});
				float stdev = sqrt(accum / (winSize - 1)); //标准差
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
	CSDistance = enhanceMatrix(CSDistance, winSize);
	GISTDistance = enhanceMatrix(GISTDistance, winSize);
	GPSDistance = enhanceMatrix(GPSDistance, winSize);
	LDBDistance = enhanceMatrix(LDBDistance, winSize);
	return true;
}

int VisualLocalization::chooseGPSDistance(double rangeTh, std::vector<std::pair<double, int>>& xGPSDistance)
{
	// 可以替换成堆排序，可降低时间复杂度到nLogK?
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

