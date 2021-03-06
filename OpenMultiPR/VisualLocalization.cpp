#include "VisualLocalization.h"
#include "Tools\Timer.h"
#include <direct.h>
#include <cmath>
#include "Tools/GNSSDistance.h"
#include "SequenceSearch.h"
#include "ParameterTuning.h"
#include "FileInterface/GroundTruth.h"
#include "Tools/dirent.h"

using namespace DBoW3;
using namespace std;


VisualLocalization::VisualLocalization(GlobalConfig& config)
{ 	
	if (!config.getValid())
	{
		throw invalid_argument("Configuration is invalid!");
	}

	this->runningMode = config.mode;

	descriptorbase = new Descriptors(config, true);
	std::cout << "--- Database Images have been processed ---" << std::endl;
	descriptorquery = new Descriptors(config, false);
	std::cout << "--- Query Images have been processed ---" << std::endl;
	matRow = descriptorquery->getVolume();
	matCol = descriptorbase->getVolume();

	this->codeBook = config.codeBook;
	cv::destroyAllWindows();
	ground.init(config.pathTest + "of.txt", config.pathRec + "of.txt");
	ground.generateGroundTruth(5);
	
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
	// 对于测试集数据和训练集数据,获取不同距离的Matrix,Matrix为empty则代表没有该项距离
	// GIST
	if ((descriptorquery->GIST)[channelIdx].empty() || (descriptorbase->GIST)[channelIdx].empty())
	{
		GISTDistance[channelIdx] = cv::Mat();
	}
	else
	{
		GISTDistance[channelIdx] = cv::Mat(matRow, matCol, CV_32FC1);
		for (size_t i = 0; i < matRow; i++)
			for (size_t j = 0; j < matCol; j++)
				GISTDistance[channelIdx].at<float>(i, j) = 
				cv::norm((descriptorquery->GIST)[channelIdx].row(i), (descriptorbase->GIST)[channelIdx].row(j), cv::NORM_L2);
	}
	// LDB
	if ((descriptorquery->LDB)[channelIdx].empty() || (descriptorbase->LDB)[channelIdx].empty())
	{
		LDBDistance[channelIdx] = cv::Mat();
	}
	else
	{
		LDBDistance[channelIdx] = cv::Mat(matRow, matCol, CV_32FC1);
		for (size_t i = 0; i < matRow; i++)
			for (size_t j = 0; j < matCol; j++)
				LDBDistance[channelIdx].at<float>(i, j) = 
				hamming_matching((descriptorquery->LDB)[channelIdx].row(i), (descriptorbase->LDB)[channelIdx].row(j));
	}

	return true;
}

bool VisualLocalization::getDistanceMatrix(float gnssTh)
{
	int matRow = descriptorquery->getVolume();
	int matCol = descriptorbase->getVolume();

	// GoogLeNet
	auto GGQuery = descriptorquery->GG;
	auto GGRef = descriptorbase->GG;
	if (GGQuery.empty()||GGRef.empty())
	{
		GGDistance = cv::Mat();
	}
	else
	{
		GGDistance = cv::Mat(matRow, matCol, CV_32FC1);
		for (size_t i = 0; i < matRow; i++)
			for (size_t j = 0; j < matCol; j++)
				GGDistance.at<float>(i, j) = cv::norm(GGQuery.row(i), GGRef.row(j), cv::NORM_L2);
	}
	// GPS
	auto GPSQuery = descriptorquery->GPS;
	auto GPSRef = descriptorbase->GPS;
	if (GPSQuery.empty() || GPSRef.empty())
	{
		GPSDistance = cv::Mat();
		GPSMask_uchar = cv::Mat();
	}
	else
	{
		GPSDistance = cv::Mat(matRow, matCol, CV_32FC1);
		for (size_t i = 0; i < matRow; i++)
			for (size_t j = 0; j < matCol; j++)
				GPSDistance.at<float>(i, j) = GNSSdistance(GPSQuery.at<float>(i,1), GPSQuery.at<float>(i, 0), GPSRef.at<float>(j, 1), GPSRef.at<float>(j, 0));
		cv::Mat GPSuchar;
		GPSDistance.convertTo(GPSuchar, CV_8U);
		cv::threshold(GPSuchar, GPSMask_uchar, gnssTh, 255, cv::THRESH_BINARY);
		//cv::threshold(GPSDistance, GPSMask, gnssTh, FLT_MAX, cv::THRESH_BINARY_INV);
		//GPSMask.convertTo(GPSMask_uchar, CV_8U);
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
	case 1: ORBQuery = std::vector<cv::Mat>(); ORBRef = std::vector<cv::Mat>(); /*BoWGlobalBest = &BoWGlobalBest_D; */ break;
	case 2: ORBQuery = descriptorquery->ORB_IR; ORBRef = descriptorbase->ORB_IR; BoWGlobalBest = &BoWGlobalBest_IR; break;
	default:
		break;
	}
	if (ORBQuery.empty() || ORBRef.empty())
	{
		if (BoWGlobalBest)
		{
			(*BoWGlobalBest) = cv::Mat();
		}		
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
			
			if (GPSMask_uchar.empty())
			{
				ORBdb.query(ORBQuery[i], ret);
				BoWGlobalBest->push_back(ret[0].Id);
			}
			else
			{
				ORBdb.query(ORBQuery[i], ret, -1);//ret 是0-based
				for (auto r : ret)
				{
					if (!GPSMask_uchar.at<uchar>(i, r.Id))
					{
						BoWGlobalBest->push_back(r.Id);
						break;
					}
				}
			}
		}
#endif // DEBUG
	}

	//cv::Mat LDBdistanceMat, GISTdistanceMat;
	std::vector<int> *LDBglobalResult = nullptr, *GISTglobalResult = nullptr;
	switch (channelIdx)
	{
	case 0: LDBglobalResult = &LDBGlobalBest_RGB; GISTglobalResult = &GISTGlobalBest_RGB; break;
	case 1: LDBglobalResult = &LDBGlobalBest_D; GISTglobalResult = &GISTGlobalBest_D; break;
	case 2: LDBglobalResult = &LDBGlobalBest_IR; GISTglobalResult = &GISTGlobalBest_IR;  break;
	default:
		break;
	}
	LDBDistance[channelIdx].setTo(FLT_MAX, GPSMask_uchar);
	GISTDistance[channelIdx].setTo(FLT_MAX, GPSMask_uchar);

	for (size_t i = 0; i < matRow; i++)//query
	{
		//When minIdx is not NULL, it must have at least 2 elements (as well as maxIdx), even if src is a single-row or single-column matrix.
		//In OpenCV (following MATLAB) each array has at least 2 dimensions, i.e. single-column matrix is Mx1 matrix (and therefore minIdx/maxIdx will be (i1,0)/(i2,0)) 
		//and single-row matrix is 1xN matrix (and therefore minIdx/maxIdx will be (0,j1)/(0,j2)).
		int* minPos = new int[2];
		cv::minMaxIdx(LDBDistance[channelIdx].row(i), nullptr, nullptr, minPos, nullptr);	//可否改成top-k		?
		LDBglobalResult->push_back(minPos[1]);
		cv::minMaxIdx(GISTDistance[channelIdx].row(i), nullptr, nullptr, minPos, nullptr);
		GISTglobalResult->push_back(minPos[1]);
		delete minPos;
	}
	return true;
}

bool VisualLocalization::getGlobalSearch()//GPS global best
{

	if (runningMode==2)
	{
		/// only GPS localization
		cv::Mat GPSdistanceMat = GPSDistance;
		GPSdistanceMat.setTo(FLT_MAX, GPSMask_uchar);
		for (size_t i = 0; i < matRow; i++)//query
		{
			//When minIdx is not NULL, it must have at least 2 elements (as well as maxIdx), even if src is a single-row or single-column matrix.
			//In OpenCV (following MATLAB) each array has at least 2 dimensions, i.e. single-column matrix is Mx1 matrix (and therefore minIdx/maxIdx will be (i1,0)/(i2,0)) 
			//and single-row matrix is 1xN matrix (and therefore minIdx/maxIdx will be (0,j1)/(0,j2)).
			int* minPos = new int[2];
			cv::minMaxIdx(GPSdistanceMat.row(i), nullptr, nullptr, minPos, nullptr);	//可否改成top-k		?
			GPSGlobalBest.push_back(minPos[1]);
			delete minPos;
		}
	}
	  
	/// GoogLeNet 
	if(GGDistance.empty())
		return true;
	//cv::Mat GGdistanceMat = GGDistance & GPSMask;
	cv::Mat GGdistanceMat = GGDistance;
	GGdistanceMat.setTo(FLT_MAX, GPSMask_uchar);

	for (size_t i = 0; i < matRow; i++)//query
	{
		//When minIdx is not NULL, it must have at least 2 elements (as well as maxIdx), even if src is a single-row or single-column matrix.
		//In OpenCV (following MATLAB) each array has at least 2 dimensions, i.e. single-column matrix is Mx1 matrix (and therefore minIdx/maxIdx will be (i1,0)/(i2,0)) 
		//and single-row matrix is 1xN matrix (and therefore minIdx/maxIdx will be (0,j1)/(0,j2)).
		int* minPos = new int[2];
		cv::minMaxIdx(GGdistanceMat.row(i), nullptr, nullptr, minPos, nullptr);	//可否改成top-k		?
		GGglobalResult.push_back(minPos[1]);
		delete minPos;
	}
	return true;
}

void VisualLocalization::getBestMatch()
{
	if (runningMode == 2)
	{
		// generate distance matric for gist \ ldb and gps && BoW 
		// get global best idx
		getDistanceMatrix((float)15);
		getGlobalSearch();
		cv::Size matSize(matCol, matRow);

		Parameter2F1 pt(ground.gt, GGglobalResult, GPSGlobalBest, /*BoWGlobalBest_D,*/ BoWGlobalBest_IR, GISTGlobalBest_RGB, GISTGlobalBest_D, GISTGlobalBest_IR,
			LDBGlobalBest_RGB, LDBGlobalBest_D, LDBGlobalBest_IR, matSize);
		float *p = new float;
		float *r = new float;
		std::cout << "GPS_TEST" << std::endl;
		std::vector<double> coeff4 = { 1,0,0, 0,0,0, 0,0,0, 0 };
		pt.updateParams(coeff4);
		pt.updateParams(0);
		pt.updateParams(1, 1, 1);
		pt.placeRecognition();
		//pt.printMatchingResults();
		std::cout << pt.calculateF1score(p, r) << "\t" << *p << "\t" << *r << std::endl;
		std::cout << pt.calculateErr() << std::endl;
		std::cout << "-----" << std::endl;
	}
	else
	{
		// generate distance matric for gist \ ldb and gps && BoW 
		// get global best idx
		getDistanceMatrix((float)15);
		for (size_t i = 0; i < 3; i++)
		{
			getDistanceMatrix((int)i);
			getGlobalSearch(i);
		}
		getGlobalSearch();
		cv::Size matSize(matCol, matRow);


		Parameter2F1 pt(ground.gt, GGglobalResult, BoWGlobalBest_RGB, /*BoWGlobalBest_D,*/ BoWGlobalBest_IR, GISTGlobalBest_RGB, GISTGlobalBest_D, GISTGlobalBest_IR,
			LDBGlobalBest_RGB, LDBGlobalBest_D, LDBGlobalBest_IR, matSize);

		// 截断
		if (runningMode == 1)
		{
			float *p = new float;
			float *r = new float;
			std::cout << "default" << std::endl;
			pt.placeRecognition();
			generateVideo(pt.getMatchingResults());
			std::cout << pt.calculateF1score(p, r) << "\t" << *p << "\t" << *r << std::endl;
			std::cout << pt.calculateErr() << std::endl;
			std::cout << "-----" << std::endl;

			//std::cout << "lambda=1" << std::endl;
			//std::vector<double> coeff3(8, 1);
			//pt.updateParams(coeff3);
			//pt.updateParams(0.16 * 9 / 10);
			//pt.placeRecognition();
			////pt.printMatchingResults();
			//std::cout << pt.calculateF1score(p, r) << "\t" << *p << "\t" << *r << std::endl;
			//std::cout << pt.calculateErr() << std::endl;
			//std::cout << "-----" << std::endl;

			//std::cout << "without GoogLeNet" << std::endl;
			//std::vector<double> coeff = 
			//{ 1.24517777,	1.685089537,
			//	1.578978596,	1.091489515,	0.987401397,
			//	0.526293315,	0.623399388,	0.840100646,
			//	1.422069836 };
			//pt.updateParams(coeff);
			//pt.updateParams(std::vector<double>(), 8, 0);
			//pt.updateParams(0.16 * 8.577930164	/ 10);
			//pt.placeRecognition();
			////pt.printMatchingResults();
			//std::cout << pt.calculateF1score(p, r) << "\t" << *p << "\t" << *r << std::endl;
			//std::cout << pt.calculateErr() << std::endl;
			//std::cout <<"-----" << std::endl;
			////
			//std::cout << "without GIST" << std::endl;
			//pt.updateParams(coeff);
			//pt.updateParams(std::vector<double>(), 3, 0);
			//pt.updateParams(std::vector<double>(), 4, 0);
			//pt.updateParams(std::vector<double>(), 2, 0);
			//pt.updateParams(0.16 * 6.342130492	/ 10);
			//pt.placeRecognition();
			////pt.printMatchingResults();
			//std::cout << pt.calculateF1score(p, r) << "\t" << *p << "\t" << *r << std::endl;
			//std::cout << pt.calculateErr() << std::endl;
			//std::cout << "-----" << std::endl;

			//std::cout << "without LDB" << std::endl;
			//pt.updateParams(coeff);
			//pt.updateParams(std::vector<double>(), 6, 0);
			//pt.updateParams(std::vector<double>(), 7, 0);
			//pt.updateParams(std::vector<double>(), 5, 0);
			//pt.updateParams(0.16 * 8.010206651	/ 10);
			//pt.placeRecognition();
			////pt.printMatchingResults();
			//std::cout << pt.calculateF1score(p, r) << "\t" << *p << "\t" << *r << std::endl;
			//std::cout << pt.calculateErr() << std::endl;
			//std::cout << "-----" << std::endl;

			//std::cout << "without BoW" << std::endl;
			//pt.updateParams(coeff);
			//pt.updateParams(std::vector<double>(), 0, 0);
			//pt.updateParams(std::vector<double>(), 1, 0);
			//pt.updateParams(0.16 * 7.069732693	/ 10);
			//pt.placeRecognition();
			////pt.printMatchingResults();
			//std::cout << pt.calculateF1score(p, r) << "\t" << *p << "\t" << *r << std::endl;
			//std::cout << pt.calculateErr() << std::endl;
			//std::cout << "-----" << std::endl;

			delete p;
			delete r;
		}

		else if (runningMode == 3)
		{
			//// use OpenGA to optimize coefficents
			std::vector<double> coeff;
			pt.prepare4MultimodalCoefficients();
			optimizeMultimodalCoefficients(&pt, coeff);
			pt.updateParams(coeff);
			////// calculate score matrix for single descriptor
			pt.placeRecognition();
			//pt.printMatchingResults();
			std::cout << pt.calculateF1score() << std::endl;
		}

		else if (runningMode == 4)
		{
			// sweep the parameter
			for (float i = 0.1; i <= 0.8; i += 0.05)
			{
				pt.updateParams(1 / i, i, 16);
				pt.placeRecognition();
				std::cout << i << "\t" << pt.calculateF1score() << std::endl;
			}
			std::cout << "\n";
			// 应该实际的值是一半
			for (float i = 3; i <= 79; i += 4)
			{
				pt.updateParams(1 / 0.4, 0.4, i);
				pt.placeRecognition();
				std::cout << i << "\t" << pt.calculateF1score() << std::endl;
			}
			std::cout << "\n";
			pt.updateParams(1 / 0.4, 0.4, 10);
			for (float i = 0; i <= 0.6; i += 0.02)
			{
				pt.updateParams(i);
				pt.placeRecognition();
				float *p = new float;
				float *r = new float;
				std::cout << i << "\t" << pt.calculateF1score(p, r) << "\t" << *p << "\t" << *r << std::endl;
				delete p;
				delete r;
			}
		}
	}


}

static bool getDistanceMap(cv::Mat DistanceMat, std::string DescriptorTtype)
{
	cv::Mat desc;
	cv::normalize(DistanceMat, desc, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	cv::resize(desc, desc, cv::Size(desc.cols * 5, desc.rows * 5));
	cv::imshow("Distance Matrix - "+ DescriptorTtype, desc);
	return true;
}

bool VisualLocalization::showDistanceMatrix()
{

	if (!LDBDistance[0].empty())
	{
		getDistanceMap(LDBDistance[0], "LDB_RGB");
	}
	if (!LDBDistance[1].empty())
	{
		getDistanceMap(LDBDistance[1], "LDB_D");
	}
	if (!LDBDistance[2].empty())
	{
		getDistanceMap(LDBDistance[2], "LDB_IR");
	}
	if (!GPSDistance.empty())
	{
		getDistanceMap(GPSDistance, "GPS");
	}
	if (!GISTDistance[0].empty())
	{
		getDistanceMap(GISTDistance[0], "GIST_RGB");
	}
	if (!GISTDistance[1].empty())
	{
		getDistanceMap(GISTDistance[1], "GIST_D");
	}
	if (!GISTDistance[2].empty())
	{
		getDistanceMap(GISTDistance[2], "GIST_IR");
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
	//CSDistance = enhanceMatrix(CSDistance, winSize);
	//GISTDistance = enhanceMatrix(GISTDistance, winSize);
	//GPSDistance = enhanceMatrix(GPSDistance, winSize);
	//LDBDistance = enhanceMatrix(LDBDistance, winSize);
	return true;
}

// 返回时间戳格式为 yyyy-mm-dd_hh-mm-ss
std::string getTimeStamp()
{
	std::time_t timep = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());	
	struct tm p;
	localtime_s(&p, &timep); /*转换为struct tm结构的当地时间*/

	stringstream timeStampStream;
	timeStampStream << 1900 + p.tm_year << setw(2) << setfill('0') << 1 + p.tm_mon << setw(2) << setfill('0') << p.tm_mday << "_";
	timeStampStream << p.tm_hour << "-" << p.tm_min << "-" << p.tm_sec;
	return timeStampStream.str();
}

bool VisualLocalization::generateVideo(std::vector<int> matchingResults, std::string path)
{
	if (path.empty())
	{
		DIR *dir;
		if ((dir = opendir("results")) == NULL)
		{
			_mkdir("results");
		}
		path = "results\\";
		path += getTimeStamp()+".avi";
	}
	cv::VideoWriter writer(path, cv::VideoWriter::fourcc('F', 'L', 'V', '1'), 1.0, cv::Size(320, 240 * 2));
	if (!writer.isOpened())
	{
		writer.release();
		return false;
	}
	for (size_t i = 0; i < matchingResults.size(); i++)
	{
		cv::Mat query_database = cv::imread(descriptorquery->picFiles.getColorImgPath(i));
		if (matchingResults[i]!=-1)
		{
			query_database.push_back(cv::imread(descriptorbase->picFiles.getColorImgPath(matchingResults[i])));
		}
		else
		{
			query_database.push_back(cv::Mat(cv::Size(320, 240), CV_8UC3, cv::Scalar(0, 0, 0)));
		}
		writer << query_database;
	}
	writer.release();
	return true;
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

