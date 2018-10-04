#include "VisualLocalization.h"
#include "Tools\Timer.h"
#include <direct.h>
#include <cmath>

using namespace DBoW3;
using namespace std;

// 返回时间戳格式为 yyyy-mm-dd_hh-mm-ss
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

VisualLocalization::VisualLocalization(GlobalConfig& config):withGPS(config.withGPS),descriptor(config.descriptor),
useDepth(config.useDepth), useIR(config.useIR), useRGBDIR(config.useRGBDIR), useRGBIR(config.useRGBIR)
{ 	
	if (!config.getValid())
	{
		throw invalid_argument("Configuration is invalid!");
	}

	descriptorbase = new Descriptorbase(config);
	descriptorquery = new DescriptorQuery(config);
	std::cout << "VisualLocalization is built! OK" << std::endl;
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

void VisualLocalization::getfStreamOut(std::string pathTest, std::ofstream* fResult, std::vector<std::string>& pathsOut, int num)
{
	string descriptorType[] = { "CS", "GIST", "LDB", "BoW", "GPS","ORB" };
	string prefix = pathTest + "\\result-";
	if (useDepth)
		prefix += "Depth";
	else if (useIR)
		prefix += "IR";
	else if (useRGBDIR)
	{
		prefix += "RGBDIR";
	}
	else if (useRGBIR)
	{
		prefix += "RGBIR";
	}
	string postfix;
	if (withGPS)
		postfix += "GPS";
	postfix += ".txt";
	for (size_t i = 0; i < num; i++)
	{
		fResult[i].open(prefix + descriptorType[i] + postfix);
		pathsOut.push_back(prefix + descriptorType[i] );
		_mkdir(pathsOut[i].c_str());
	}
	
}

void VisualLocalization::getGISTDistance(cv::Mat& GISTDistance)
{
	int matRow = descriptorquery->getVolume();
	int matCol = descriptorbase->getVolume();

	auto GISTQuery = descriptorquery->getGISTQuery();
	auto GISTRec = descriptorbase->xGISTRec;

	if (GISTQuery.empty() || GISTRec.empty())
	{
		GISTDistance = cv::Mat();
	}
	else
	{
		GISTDistance = cv::Mat(matRow, matCol, CV_64FC1);
		for (size_t i = 0; i < matRow; i++)
			for (size_t j = 0; j < matCol; j++)
				GISTDistance.at<double>(i, j) = cv::norm(cv::Mat(GISTQuery[i]), cv::Mat(GISTRec[j]), cv::NORM_L2);
	}
}

void VisualLocalization::getLDBDistance(cv::Mat& LDBDistance)
{
	int matRow = descriptorquery->getVolume();
	int matCol = descriptorbase->getVolume();

	auto LDBQuery = descriptorquery->getLDBQuery();
	auto LDBRec = descriptorbase->xLDBRec;

	if (LDBQuery.empty() || LDBRec.empty())
	{
		LDBDistance = cv::Mat();
	}
	else
	{
		LDBDistance = cv::Mat(matRow, matCol, CV_64FC1);
		for (size_t i = 0; i < matRow; i++)
			for (size_t j = 0; j < matCol; j++)
				LDBDistance.at<double>(i, j) = hamming_matching(LDBQuery.row(i), LDBRec.row(j));
	}
}

void VisualLocalization::getGPSDistance(cv::Mat& GPSDistance)
{
	auto GPSQuery = descriptorquery->getGPSQuery();
	auto GPSRec = descriptorbase->xGPSRec;

	int matRow = descriptorquery->getVolume();
	int matCol = descriptorbase->getVolume();

	if (GPSQuery.empty() || GPSRec.empty())
	{
		GPSDistance = cv::Mat();
	}
	else
	{
		GPSDistance = cv::Mat(matRow, matCol, CV_64FC1);
		for (size_t i = 0; i < matRow; i++)
			for (size_t j = 0; j < matCol; j++)
				GPSDistance.at<double>(i, j) = sqrt(std::pow((GPSQuery[i].first - GPSRec[j].first), 2) + std::pow((GPSQuery[i].second - GPSRec[j].second), 2));
	}
}

void VisualLocalization::getORBDistance(cv::Mat& ORBDistance)
{
	/*ORB*/
	auto ORBRec = descriptorbase->xORBRec;
	auto ORBQuery = descriptorquery->getORBQuery();

	int matRow = descriptorquery->getVolume();
	int matCol = descriptorbase->getVolume();
	
	if (ORBQuery.empty() || ORBRec.empty())
	{
		ORBDistance = cv::Mat();
	}
	else
	{
		ORBDistance = cv::Mat(matRow, matCol, CV_64FC1);
		//Matching descriptor vectors with a brute force matcher
		//创建一个BFMatcher匹配器，BFMatcher类构造函数如下：两个参数都有默认值，但是第一个距离类型下面使用的并不是默认值，而是汉明距离
		cv::BFMatcher matcher(cv::NORM_HAMMING);
		for (size_t i = 0; i < matRow; i++)
		{
			for (size_t j = 0; j < matCol; j++)
			{
				//创建一个匹配点数组，用于承接匹配出的DMatch，其实叫match_points_array更为贴切。matches类型为数组，元素类型为DMatch
				std::vector<cv::DMatch> matches;
				matcher.match(ORBQuery.row(i), ORBRec[j], matches);
				//遍历matches[]数组，找出匹配点的最大距离和最小距离，用于后面的匹配点筛选。
				//这里的距离是上方求出的汉明距离数组，汉明距离表征了两个匹配的相似程度，所以也就找出了最相似和最不相似的两组点之间的距离。
				double min_dist = 10000, max_dist = 0;//定义距离

				for (int i = 0; i < ORBQuery.row(i).rows; ++i)//遍历
				{
					double dist = matches[i].distance;
					if (dist<min_dist) min_dist = dist;
					if (dist>max_dist) max_dist = dist;
				}

				//根据最小距离，对匹配点进行筛选，
				//当描述自之间的距离大于两倍的min_dist，即认为匹配有误，舍弃掉。
				//但是有时最小距离非常小，比如趋近于0了，所以这样就会导致min_dist到2*min_dist之间没有几个匹配。
				// 所以，在2*min_dist小于30的时候，就取30当上限值，小于30即可，不用2*min_dist这个值了
				//std::vector<cv::DMatch> good_matches;
				int pairNum = 0;
				for (int j = 0; j < ORBQuery.row(i).rows; ++j)
				{
					if (matches[j].distance <= max(1000 * min_dist, 30.0))
						//good_matches.push_back(matches[j]);
						pairNum++;
				}

				ORBDistance.at<double>(i, j) = pairNum;
			}
		}

				

		
		//////////////////////////// 使用RANSAC方法进行匹配点筛选
		//std::vector<cv::Point2f> queryMatches, trainMatches;
		//for (auto i = 0; i < matches.size(); i++)
		//{
		//	queryMatches.push_back(keypoints_1[matches[i].queryIdx].pt);
		//	trainMatches.push_back(keypoints_2[matches[i].trainIdx].pt);
		//}

		//std::vector<uchar> matchStatus;
		//RansacAffine(queryMatches, trainMatches, matchStatus, 10);

		//int distanceScore = 0;
		//std::vector<DMatch> bestMatches;
		////    bestMatches = ransac(keypoints_2, keypoints_1, matches);
		//for (auto i = 0; i < matchStatus.size(); i++)
		//{
		//	if (matchStatus[i] != 0)
		//	{
		//		bestMatches.push_back(matches[i]);
		//		distanceScore += 1;
		//	}
		//}

	}
	

	//std::cout << distGPS << std::endl;
	//if (distGPS > 0.014)
	//	continue;

	//double distCS = arma::norm(xCS - xCSRec[i], 2);
	//xCSDistance.push_back(std::pair<double, int>(distCS, i));

	//vector<DMatch> matches;
	//matcher->match(xSURFRec[i], descriptorsSURF, matches);
	//for each (auto var in matches)
	//{
	//	var.
	//}
	//xSURFDistance.push_back(
	//排序之前，对于距离数据先行输出以作拟合用
	//照片距离
	//fout << abs(distanceBase[i] - distanceBase[picsTest.bestMatchValue]) << "\t"<< distGPS << "\t" << distGIST << "\t" << distLDB << std::endl;

}

bool VisualLocalization::test(GlobalConfig & config, const bool& outIdxDev, const bool &outPR, double* PandR, int kLDB, int kGIST, int kBoW, double r, double numKeyRatio, double numKeyTh)
{
	// 输出CS GIST LDB BoW GPS特征匹配误差的文件流
	string timeStamp = getTimeStamp();
	// 创建时间戳目录,保存匹配结果
	string content = descriptorquery->pathsTest[0] + "\\" + timeStamp;
	_mkdir(content.c_str());
	ofstream fResult[6];
	vector<string> fOutPath;
	if (outIdxDev)
	{
		getfStreamOut(content, fResult, fOutPath, 6);
	}
	// 输出关键点识别P&R的文件流
	ofstream fout;
	if (outPR)
	{
		fout.open(content +"\\ofout.txt");
	}

	// 对于测试集数据和训练集数据,获取不同距离的Matrix,Matrix为empty则代表没有该项距离
	cv::Mat GISTDistance;
	cv::Mat LDBDistance;
	cv::Mat GPSDistance;
	cv::Mat ORBMatchingPairNum;
	getGISTDistance(GISTDistance);
	getLDBDistance(LDBDistance);
	getGPSDistance(GPSDistance);
	getORBDistance(ORBMatchingPairNum);

	Timer timer;
	timer.start();

	// calculate the difference of tested img to all recorded images (except for CS) 计算特征包括在内，距离没有排序
	//std::vector<std::pair<double, int>> xCSDistance;
	QueryResults ret;

	int querylength = descriptorquery->getVolume();
	int baselength = descriptorbase->getVolume();
	for (size_t i = 0; i < querylength; i++)
	{
		std::vector<std::pair<double, int>> xGISTDistance;
		std::vector<std::pair<double, int>> xLDBDistance;
		std::vector<std::pair<double, int>> xGPSDistance;
		std::vector<std::pair<double, int>> xORBPairNum;
		for (size_t j = 0; j < baselength; j++)
		{
			xGISTDistance.push_back(std::pair<double, int>(GISTDistance.at<double>(i, j), j));
			xLDBDistance.push_back(std::pair<double, int>(LDBDistance.at<double>(i, j), j));
			xGPSDistance.push_back(std::pair<double, int>(GPSDistance.at<double>(i, j), j));
			xORBPairNum.push_back(std::pair<double, int>(ORBMatchingPairNum.at<double>(i, j), j));
		}

		//std::sort(xCSDistance.begin(), xCSDistance.end(), cmp);
		std::sort(xGISTDistance.begin(), xGISTDistance.end(), cmp);
		std::sort(xLDBDistance.begin(), xLDBDistance.end(), cmp);
		std::sort(xORBPairNum.begin(), xORBPairNum.end(), cmpPairNum);

		int GPSrear = querylength;
		if (descriptor == "GPS" || withGPS)
		{
			GPSrear = chooseGPSDistance(r, xGPSDistance); //改用张数限制如何？

			std::cout << GPSrear << std::endl;
		}
		/*进行GPS范围内的图像匹配结果确定-5NN*/
		//std::vector<std::pair<double, int>> retCS;//distance，idx
		//getMatchingResult(xGPSDistance, GPSrear, xCSDistance, 5, retCS);
		std::vector<std::pair<double, int>> retGIST;//distance，idx
		getMatchingResult(xGPSDistance, GPSrear, xGISTDistance, kGIST, retGIST);
		std::vector<std::pair<double, int>> retLDB;//distance
		getMatchingResult(xGPSDistance, GPSrear, xLDBDistance, kLDB, retLDB);
		std::vector<std::pair<double, int>> retBoW;//相关程度(0~1)
		getMatchingResult(xGPSDistance, GPSrear, ret, kBoW, retBoW);
		std::vector<std::pair<double, int>> retORB;//匹配点数
		getMatchingResult(xGPSDistance, GPSrear, xORBPairNum, 1, retORB);
		// 记录预测的library index值以及次数（关键点实际上是一点）
		std::map<int, int> timesPerPosition;
		for (size_t i = 0; i < retGIST.size(); i++)
		{
			if (!descriptorbase->isLabel[retGIST[i].second])
			{
				continue;
			}
			if (timesPerPosition.count(descriptorbase->distanceBase[retGIST[i].second]))
			{
				timesPerPosition[descriptorbase->distanceBase[retGIST[i].second]] += 1;
			}
			else
			{
				timesPerPosition[descriptorbase->distanceBase[retGIST[i].second]] = 1;
			}
		}
		for (size_t i = 0; i < retLDB.size(); i++)
		{
			if (!descriptorbase->isLabel[retLDB[i].second])
			{
				continue;
			}
			if (timesPerPosition.count(descriptorbase->distanceBase[retLDB[i].second]))
			{
				timesPerPosition[descriptorbase->distanceBase[retLDB[i].second]] += 1;
			}
			else
			{
				timesPerPosition[descriptorbase->distanceBase[retLDB[i].second]] = 1;
			}
		}
		for (size_t i = 0; i < retBoW.size(); i++)
		{
			if (!descriptorbase->isLabel[retBoW[i].second])
			{
				continue;
			}
			if (timesPerPosition.count(descriptorbase->distanceBase[retBoW[i].second]))
			{
				timesPerPosition[descriptorbase->distanceBase[retBoW[i].second]] += 1;
			}
			else
			{
				timesPerPosition[descriptorbase->distanceBase[retBoW[i].second]] = 1;
			}
		}
		//把map中元素转存到vector中   
		vector<pair<int, int>> timesPrePosition_vec(timesPerPosition.begin(), timesPerPosition.end());
		int numKey = 0;
		if (!timesPrePosition_vec.empty())
		{
			sort(timesPrePosition_vec.begin(), timesPrePosition_vec.end(), cmpSec);
			numKey = timesPrePosition_vec.back().second;
			//输出结果
			std::cout << descriptorquery->picsTest->posLabelValue << std::endl;
			for (size_t i = 0; i <timesPrePosition_vec.size(); i++)
			{
				std::cout << timesPrePosition_vec[i].first << "\t";
				std::cout << timesPrePosition_vec[i].second << "\t";

			}
			std::cout << std::endl;
			//waitKey(0);
		}
		// key position prediction 
		int all = 0;
		for (auto & var : timesPerPosition)
		{
			all += var.second;
		}

		//retBoW.size() + retLDB.size() + retGIST.size();
		keyPredict.push_back(numKey >= max(all*numKeyRatio, numKeyTh));//3 for color
																	   // GPS prediction ??
																	   //int numKeyGPS = 0;
																	   //for (size_t i = 0; i < 20; i++)
																	   //{
																	   //	if (isLabel[xGPSDistance[i].second])
																	   //	{
																	   //		numKeyGPS++;
																	   //	}
																	   //}
		keyGPS.push_back(descriptorbase->isLabel[xGPSDistance[0].second]);
		if (outPR&&outIdxDev)
		{
			keyGT.push_back(descriptorbase->isLabel[descriptorquery->picsTest->bestMatchValue]);
			fout << descriptorbase->isLabel[descriptorquery->picsTest->bestMatchValue] << "\t" << keyPredict.back() << "\t" << keyGPS.back() << std::endl;//2
		}
		else if (outPR)
		{
			keyGT.push_back(descriptorquery->picsTest->posLabelValue);
			fout << descriptorquery->picsTest->posLabelValue << "\t" << keyPredict.back() << "\t" << keyGPS.back() << std::endl;
		}

		//double sumGIST = 0;
		//double wsum = 1e-7; 
		//for (size_t i = 0; i < retGIST.size(); i++)
		//{
		//	sumGIST += (1.0 / retGIST[i].first)*retGIST[i].second;
		//	wsum += (1.0 / retGIST[i].first);
		//}
		//double GISTidx = sumGIST / wsum;
		//double sumLDB = 0; wsum = 1e-7;
		//for (size_t i = 0; i < retLDB.size(); i++)
		//{
		//	sumLDB += (1.0 / retLDB[i].first)*retLDB[i].second;
		//	wsum += (1.0 / retLDB[i].first);
		//}
		//double LDBidx = sumLDB / wsum;
		//double sumBoW = 0; wsum = 1e-7;
		//for (size_t i = 0; i < retBoW.size(); i++)
		//{
		//	sumBoW += ( retBoW[i].first)*retBoW[i].second;
		//	wsum += ( retBoW[i].first);
		//}
		//double BoWidx = sumBoW / wsum;
		//std::cout << retGIST.size()<<"\t"<<GISTidx <<"\t"<<retLDB.size() << "\t" << LDBidx << "\t" << retBoW.size() << "\t" << BoWidx << std::endl;
		//int idxRet = 0.5*GISTidx + 0.3*BoWidx + 0.2*LDBidx/(0.5*(sumGIST!=0)+0.3*(sumBoW != 0) +0.2*(sumLDB != 0));
		//if (retBoW.empty()&& retLDB.empty()&& retGIST.empty())
		//{
		//	fout<< abs(distanceBase[xGPSDistance[0].second] - distanceBase[picsTest.bestMatchValue]) << std::endl;
		//}
		//else
		//{
		//	fout << abs(distanceBase[idxRet] - distanceBase[picsTest.bestMatchValue]) << std::endl;
		//}

		// CS GIST LDB BoW GPS 输出单个特征的误差，保存到文件
		if (outIdxDev)
		{
			//outputRet(fResult[0], retCS, picsTest.bestMatchValue, 5, fOutPath[0], picsTest.getFilePointer());
			outputRet(fResult[1], retGIST, descriptorquery->picsTest->bestMatchValue, 5, fOutPath[1], descriptorquery->picsTest->getFilePointer());
			outputRet(fResult[2], retLDB, descriptorquery->picsTest->bestMatchValue, 5, fOutPath[2], descriptorquery->picsTest->getFilePointer());
			outputRet(fResult[3], retBoW, descriptorquery->picsTest->bestMatchValue, 5, fOutPath[3], descriptorquery->picsTest->getFilePointer());
			outputRet(fResult[4], xGPSDistance, descriptorquery->picsTest->bestMatchValue, fOutPath[4], descriptorquery->picsTest->getFilePointer());
		}


			//fResult[5] << i << "\t" << retORB[i].second << std::endl;

		//cv::waitKey(0);	// Waiting for key pressed.

	}

		


	// 记录p&r
	if (outPR)
	{
		double precisionGPS = 0, precisionVL = 0;
		double recallGPS = 0, recallVL = 0;
		double tpVL = 0, fpVL = 0, fnVL = 0, tpGPS = 0, fpGPS = 0, fnGPS = 0;
		for (size_t i = 0; i < keyGT.size(); i++)
		{
			//visual localization
			if ((keyGT[i]==true)&&keyPredict[i]==true)
			{
				tpVL++;
			}
			else if(keyGT[i] == true && keyPredict[i] == false)
			{
				fnVL++;
			}
			else if (keyGT[i] == false && keyPredict[i] == true)
			{
				fpVL++;
			}
			//GPS
			if (keyGT[i] == true && keyGPS[i] == true)
			{
				tpGPS++;
			}
			else if (keyGT[i] == true && keyGPS[i] == false)
			{
				fnGPS++;
			}
			else if (keyGT[i] == false && keyGPS[i] == true)
			{
				fpGPS++;
			}
		}
		precisionVL = tpVL / (tpVL + fpVL);
		recallVL = tpVL /(tpVL +fnVL);
		precisionGPS = tpGPS / (tpGPS + fpGPS);
		recallGPS = tpGPS / (tpGPS + fnGPS);
		std::cout << precisionVL << "\t"<<recallVL<<"\t"<<precisionGPS << "\t" << recallGPS;
		PandR[0] = precisionVL;
		PandR[1] = recallVL;
		PandR[2] = precisionGPS;
		PandR[3] = recallGPS;

			//getchar();
	}

	return true;
}

void VisualLocalization::getMatchingResultfromDistance(const cv::Mat& distance, std::vector<std::pair<double, int>>& outResult, const int& i)
{
	std::vector<std::pair<double, int>> xDistance;
	int baselength = descriptorbase->getVolume();
	for (size_t j = 0; j < baselength; j++)
	{
		xDistance.push_back(std::pair<double, int>(distance.at<double>(i, j), j));
	}
	std::sort(xDistance.begin(), xDistance.end(), cmp);
}

void VisualLocalization::getMatchingResultfromPair(const cv::Mat& in, std::vector<std::pair<double, int>>& outResult)
{

}

bool VisualLocalization::GPSLocalize()
{
	string timeStamp = getTimeStamp();
	// 创建时间戳目录,保存匹配结果
	string content = descriptorquery->pathsTest[0] + "\\" + timeStamp;
	_mkdir(content.c_str());
	ofstream fResult(content + "\\result-GPS.txt");


	// 对于测试集数据和训练集数据,获取不同距离的Matrix,Matrix为empty则代表没有该项距离
	cv::Mat GPSDistance;
	getGPSDistance(GPSDistance);

	Timer timer;
	timer.start();

	// calculate the difference of tested img to all recorded images (except for CS) 计算特征包括在内，距离没有排序
	//std::vector<std::pair<double, int>> xCSDistance;
	QueryResults ret;

	int querylength = descriptorquery->getVolume();
	int baselength = descriptorbase->getVolume();
	for (size_t i = 0; i < querylength; i++)
	{
		std::vector<std::pair<double, int>> xGPSDistance;
		for (size_t j = 0; j < baselength; j++)
		{
			xGPSDistance.push_back(std::pair<double, int>(GPSDistance.at<double>(i, j), j));
		}

		std::sort(xGPSDistance.begin(), xGPSDistance.end(), cmp);
		
		fResult << i << "\t" << xGPSDistance[0].second << std::endl;

	}

	return true;
}

bool VisualLocalization::paperLocalize(const bool & outIdxErr, double* PandR, const bool & outPR, int kLDB, int kGIST, int kBoW, double r, double numKeyRatio, double numKeyTh)
{
	// 输出CS GIST LDB BoW GPS特征匹配误差的文件流
	string timeStamp = getTimeStamp();
	// 创建时间戳目录
	string content = descriptorquery->pathsTest[0] + "\\" + timeStamp;
	_mkdir(content.c_str());
	
	// 输出关键点识别P&R的文件流
	ofstream fout;
	if (outPR)
	{
		// fout.open(content + "\\ofout.txt");
		fout.open(content + "\\KeyPositionP&R.txt");
	}


	// 对于测试集数据和训练集数据,获取不同距离的Matrix,Matrix为empty则代表没有该项距离
	cv::Mat GISTDistance;
	cv::Mat LDBDistance;
	cv::Mat GPSDistance;
	getGISTDistance(GISTDistance);
	getLDBDistance(LDBDistance);
	getGPSDistance(GPSDistance);

	Timer timer;
	timer.start();

	// calculate the difference of tested img to all recorded images 计算特征包括在内，距离没有排序
	QueryResults ret;

	int querylength = descriptorquery->getVolume();
	int baselength = descriptorbase->getVolume();
	for (size_t i = 0; i < querylength; i++)
	{
		std::vector<std::pair<double, int>> xGISTDistance;
		std::vector<std::pair<double, int>> xLDBDistance;
		std::vector<std::pair<double, int>> xGPSDistance;
		for (size_t j = 0; j < baselength; j++)
		{
			xGISTDistance.push_back(std::pair<double, int>(GISTDistance.at<double>(i, j), j));
			xLDBDistance.push_back(std::pair<double, int>(LDBDistance.at<double>(i, j), j));
			xGPSDistance.push_back(std::pair<double, int>(GPSDistance.at<double>(i, j), j));
		}

		std::sort(xGISTDistance.begin(), xGISTDistance.end(), cmp);
		std::sort(xLDBDistance.begin(), xLDBDistance.end(), cmp);

		int GPSrear = querylength;
		if (descriptor == "GPS" || withGPS)
		{
			GPSrear = chooseGPSDistance(r, xGPSDistance); //改用张数限制如何？
			//std::cout << GPSrear << std::endl;
		}
		/*进行GPS范围内的图像匹配结果确定-5NN*/
		std::vector<std::pair<double, int>> retGIST;//distance，idx
		getMatchingResult(xGPSDistance, GPSrear, xGISTDistance, kGIST, retGIST);
		std::vector<std::pair<double, int>> retLDB;//distance
		getMatchingResult(xGPSDistance, GPSrear, xLDBDistance, kLDB, retLDB);
		std::vector<std::pair<double, int>> retBoW;//相关程度(0~1)
		getMatchingResult(xGPSDistance, GPSrear, ret, kBoW, retBoW);
		// 记录预测的library index值以及次数（关键点实际上是一点）
		std::map<int, int> timesPerPosition;
		for (size_t i = 0; i < retGIST.size(); i++)
		{
			if (!descriptorbase->isLabel[retGIST[i].second])
			{
				continue;
			}
			if (timesPerPosition.count(descriptorbase->distanceBase[retGIST[i].second]))
			{
				timesPerPosition[descriptorbase->distanceBase[retGIST[i].second]] += 1;
			}
			else
			{
				timesPerPosition[descriptorbase->distanceBase[retGIST[i].second]] = 1;
			}
		}
		for (size_t i = 0; i < retLDB.size(); i++)
		{
			if (!descriptorbase->isLabel[retLDB[i].second])
			{
				continue;
			}
			if (timesPerPosition.count(descriptorbase->distanceBase[retLDB[i].second]))
			{
				timesPerPosition[descriptorbase->distanceBase[retLDB[i].second]] += 1;
			}
			else
			{
				timesPerPosition[descriptorbase->distanceBase[retLDB[i].second]] = 1;
			}
		}
		for (size_t i = 0; i < retBoW.size(); i++)
		{
			if (!descriptorbase->isLabel[retBoW[i].second])
			{
				continue;
			}
			if (timesPerPosition.count(descriptorbase->distanceBase[retBoW[i].second]))
			{
				timesPerPosition[descriptorbase->distanceBase[retBoW[i].second]] += 1;
			}
			else
			{
				timesPerPosition[descriptorbase->distanceBase[retBoW[i].second]] = 1;
			}
		}
		//把map中元素转存到vector中   
		vector<pair<int, int>> timesPrePosition_vec(timesPerPosition.begin(), timesPerPosition.end());
		int numKey = 0;
		if (!timesPrePosition_vec.empty())
		{
			sort(timesPrePosition_vec.begin(), timesPrePosition_vec.end(), cmpSec);
			numKey = timesPrePosition_vec.back().second;
			//输出结果
			std::cout << descriptorquery->picsTest->posLabelValue << "\t";
			for (size_t i = 0; i <timesPrePosition_vec.size(); i++)
			{
				std::cout << timesPrePosition_vec[i].first << "\t";
				std::cout << timesPrePosition_vec[i].second << "\t";
			}
			std::cout << std::endl;
		}
		// key position prediction 
		int all = 0;
		for (auto & var : timesPerPosition)
		{
			all += var.second;
		}

		//retBoW.size() + retLDB.size() + retGIST.size();
		keyPredict.push_back(numKey > max(all*numKeyRatio, numKeyTh));//3 for color
																	   // GPS prediction ??
																	   //int numKeyGPS = 0;
																	   //for (size_t i = 0; i < 20; i++)
																	   //{
																	   //	if (isLabel[xGPSDistance[i].second])
																	   //	{
																	   //		numKeyGPS++;
																	   //	}
																	   //}
		keyGPS.push_back(descriptorbase->isLabel[xGPSDistance[0].second]);
		if (outPR&&outIdxErr)
		{
			keyGT.push_back(descriptorbase->isLabel[descriptorquery->picsTest->getBestMatch(i)]);
			fout << descriptorbase->isLabel[descriptorquery->picsTest->getBestMatch(i)] << "\t" << keyPredict.back() << "\t" << keyGPS.back() << std::endl;//2
		}
		if (outPR)
		{
			keyGT.push_back(descriptorquery->picsTest->getPosLabel(i));
			fout << descriptorquery->picsTest->getPosLabel(i) << "\t" << keyPredict.back() << "\t" << keyGPS.back() << std::endl;//2
		}

		//double sumGIST = 0;
		//double wsum = 1e-7;
		//for (size_t i = 0; i < retGIST.size(); i++)
		//{
		//	sumGIST += (1.0 / retGIST[i].first)*retGIST[i].second;
		//	wsum += (1.0 / retGIST[i].first);
		//}
		//double GISTidx = sumGIST / wsum;
		//double sumLDB = 0; wsum = 1e-7;
		//for (size_t i = 0; i < retLDB.size(); i++)
		//{
		//	sumLDB += (1.0 / retLDB[i].first)*retLDB[i].second;
		//	wsum += (1.0 / retLDB[i].first);
		//}
		//double LDBidx = sumLDB / wsum;
		//double sumBoW = 0; wsum = 1e-7;
		//for (size_t i = 0; i < retBoW.size(); i++)
		//{
		//	sumBoW += ( retBoW[i].first)*retBoW[i].second;
		//	wsum += ( retBoW[i].first);
		//}
		//double BoWidx = sumBoW / wsum;
		//std::cout << retGIST.size()<<"\t"<<GISTidx <<"\t"<<retLDB.size() << "\t" << LDBidx << "\t" << retBoW.size() << "\t" << BoWidx << std::endl;
		//int idxRet = 0.5*GISTidx + 0.3*BoWidx + 0.2*LDBidx/(0.5*(sumGIST!=0)+0.3*(sumBoW != 0) +0.2*(sumLDB != 0));
		//if (retBoW.empty()&& retLDB.empty()&& retGIST.empty())
		//{
		//	fout<< abs(distanceBase[xGPSDistance[0].second] - distanceBase[picsTest.bestMatchValue]) << std::endl;
		//}
		//else
		//{
		//	fout << abs(distanceBase[idxRet] - distanceBase[picsTest.bestMatchValue]) << std::endl;
		//}
	}

	// 记录p&r
	if (outPR)
	{
		double precisionGPS = 0, precisionVL = 0;
		double recallGPS = 0, recallVL = 0;
		double tpVL = 0, fpVL = 0, fnVL = 0, tpGPS = 0, fpGPS = 0, fnGPS = 0;
		for (size_t i = 0; i < keyGT.size(); i++)
		{
			//visual localization
			if ((keyGT[i] == true) && keyPredict[i] == true)
			{
				tpVL++;
			}
			else if (keyGT[i] == true && keyPredict[i] == false)
			{
				fnVL++;
			}
			else if (keyGT[i] == false && keyPredict[i] == true)
			{
				fpVL++;
			}
			//GPS
			if (keyGT[i] == true && keyGPS[i] == true)
			{
				tpGPS++;
			}
			else if (keyGT[i] == true && keyGPS[i] == false)
			{
				fnGPS++;
			}
			else if (keyGT[i] == false && keyGPS[i] == true)
			{
				fpGPS++;
			}
		}
		precisionVL = tpVL / (tpVL + fpVL);
		recallVL = tpVL / (tpVL + fnVL);
		precisionGPS = tpGPS / (tpGPS + fpGPS);
		recallGPS = tpGPS / (tpGPS + fnGPS);
		std::cout << precisionVL << "\t" << recallVL << "\t" << precisionGPS << "\t" << recallGPS;
		PandR[0] = precisionVL;
		PandR[1] = recallVL;
		PandR[2] = precisionGPS;
		PandR[3] = recallGPS;

		//getchar();
	}

	return true;
}


//bool VisualLocalization::preparePara()
//{
//	std::vector<std::string> pathsTest;
//	pathsTest.push_back(pathTest);
//	PicGnssFile picsTest(pathsTest, PicGnssFile::RGBDIR, true, 1);
//
//	// 对于测试集数据
//	//std::vector<std::vector<std::pair<double, int>>> gist;
//	//std::vector<std::vector<std::pair<double, int>>> ldb;
//	//std::vector<std::vector<std::pair<double, int>>> gps;
//	while (picsTest.doMain())
//	{
//		int num = descriptorbase->getVolume();
//		// calculate the difference of tested img to all recorded images (except for CS) 计算特征包括在内，距离没有排序
//		std::vector<std::pair<double, int>> xCSDistance;
//		std::vector<std::pair<double, int>> xGISTDistance;
//		std::vector<std::pair<double, int>> xLDBDistance;
//		std::vector<std::pair<double, int>> xGPSDistance;
//		QueryResults ret;
//		getDistance(num, picsTest, xGISTDistance, xLDBDistance, xCSDistance, xGPSDistance, ret);
//		//std::sort(xCSDistance.begin(), xCSDistance.end(), cmp);
//		std::sort(xGISTDistance.begin(), xGISTDistance.end(), cmp);
//		std::sort(xLDBDistance.begin(), xLDBDistance.end(), cmp);
//		gist.push_back(xGISTDistance);
//		ldb.push_back(xLDBDistance);
//		gps.push_back(xGPSDistance);
//		bow.push_back(ret);
//
//		keyGT.push_back(picsTest.posLabelValue);
//			//fout << picsTest.posLabelValue << "\t" << keyPredict.back() << "\t" << keyGPS.back() << std::endl;
//		
//	}
//	return true;
//
//}
//bool VisualLocalization::testPara(const bool& outIdxDev, const bool &outPR, double* PandR, int kLDB, int kGIST, int kBoW, double r, double numKeyRatio, double numKeyTh)
//{
//	keyPredict.clear();
//	keyGPS.clear(); //真值，预测，GPS
//	for (size_t i = 0; i < gist.size(); i++)
//	{
//		std::vector<std::pair<double, int>> xGISTDistance = gist[i];
//		std::vector<std::pair<double, int>> xLDBDistance = ldb[i];
//		std::vector<std::pair<double, int>> xGPSDistance = gps[i];
//		QueryResults ret = bow[i];
//		int GPSrear;
//		if (descriptor == "GPS" || withGPS)
//		{
//			GPSrear = chooseGPSDistance(r, xGPSDistance); //改用张数限制如何？
//
//			std::cout << GPSrear << std::endl;
//		}
//		/*进行GPS范围内的图像匹配结果确定-5NN*/
//		//std::vector<std::pair<double, int>> retCS;//distance，idx
//		//getMatchingResult(xGPSDistance, GPSrear, xCSDistance, 5, retCS);
//		std::vector<std::pair<double, int>> retGIST;//distance，idx
//		getMatchingResult(xGPSDistance, GPSrear, xGISTDistance, kGIST, retGIST);
//		std::vector<std::pair<double, int>> retLDB;//distance
//		getMatchingResult(xGPSDistance, GPSrear, xLDBDistance, kLDB, retLDB);
//		std::vector<std::pair<double, int>> retBoW;//相关程度(0~1)
//		getMatchingResult(xGPSDistance, GPSrear, ret, kBoW, retBoW);
//		// 记录预测的library index值以及次数（关键点实际上是一点）
//		std::map<int, int> timesPerPosition;
//		for (size_t i = 0; i < retGIST.size(); i++)
//		{
//			if (!isLabel[retGIST[i].second])
//			{
//				continue;
//			}
//			if (timesPerPosition.count(distanceBase[retGIST[i].second]))
//			{
//				timesPerPosition[distanceBase[retGIST[i].second]] += 1;
//			}
//			else
//			{
//				timesPerPosition[distanceBase[retGIST[i].second]] = 1;
//			}
//		}
//		for (size_t i = 0; i < retLDB.size(); i++)
//		{
//			if (!isLabel[retLDB[i].second])
//			{
//				continue;
//			}
//			if (timesPerPosition.count(distanceBase[retLDB[i].second]))
//			{
//				timesPerPosition[distanceBase[retLDB[i].second]] += 1;
//			}
//			else
//			{
//				timesPerPosition[distanceBase[retLDB[i].second]] = 1;
//			}
//		}
//		for (size_t i = 0; i < retBoW.size(); i++)
//		{
//			if (!isLabel[retBoW[i].second])
//			{
//				continue;
//			}
//			if (timesPerPosition.count(distanceBase[retBoW[i].second]))
//			{
//				timesPerPosition[distanceBase[retBoW[i].second]] += 1;
//			}
//			else
//			{
//				timesPerPosition[distanceBase[retBoW[i].second]] = 1;
//			}
//		}
//		//把map中元素转存到vector中   
//		vector<pair<int, int>> timesPrePosition_vec(timesPerPosition.begin(), timesPerPosition.end());
//		int numKey = 0;
//		if (!timesPrePosition_vec.empty())
//		{
//			sort(timesPrePosition_vec.begin(), timesPrePosition_vec.end(), cmpSec);
//			numKey = timesPrePosition_vec.back().second;
//		}
//		// key position prediction 
//		int all = 0;
//		for each (auto var in timesPerPosition)
//		{
//			all += var.second;
//		}
//
//		//retBoW.size() + retLDB.size() + retGIST.size();
//		keyPredict.push_back(numKey >= max(all*numKeyRatio, numKeyTh));//3 for color
//																	   // GPS prediction ??
//																	   //int numKeyGPS = 0;
//																	   //for (size_t i = 0; i < 20; i++)
//																	   //{
//																	   //	if (isLabel[xGPSDistance[i].second])
//																	   //	{
//																	   //		numKeyGPS++;
//																	   //	}
//																	   //}
//		keyGPS.push_back(isLabel[xGPSDistance[0].second]);
//
//
//	}
//	// 记录p&r
//	if (outPR)
//	{
//		double precisionGPS = 0, precisionVL = 0;
//		double recallGPS = 0, recallVL = 0;
//		double tpVL = 0, fpVL = 0, fnVL = 0, tpGPS = 0, fpGPS = 0, fnGPS = 0;
//		for (size_t i = 0; i < keyGT.size(); i++)
//		{
//			//visual localization
//			if ((keyGT[i] == true) && keyPredict[i] == true)
//			{
//				tpVL++;
//			}
//			else if (keyGT[i] == true && keyPredict[i] == false)
//			{
//				fnVL++;
//			}
//			else if (keyGT[i] == false && keyPredict[i] == true)
//			{
//				fpVL++;
//			}
//			//GPS
//			if (keyGT[i] == true && keyGPS[i] == true)
//			{
//				tpGPS++;
//			}
//			else if (keyGT[i] == true && keyGPS[i] == false)
//			{
//				fnGPS++;
//			}
//			else if (keyGT[i] == false && keyGPS[i] == true)
//			{
//				fpGPS++;
//			}
//		}
//		precisionVL = tpVL / (tpVL + fpVL);
//		recallVL = tpVL / (tpVL + fnVL);
//		precisionGPS = tpGPS / (tpGPS + fpGPS);
//		recallGPS = tpGPS / (tpGPS + fnGPS);
//		std::cout << precisionVL << "\t" << recallVL << "\t" << precisionGPS << "\t" << recallGPS;
//		PandR[0] = precisionVL;
//		PandR[1] = recallVL;
//		PandR[2] = precisionGPS;
//		PandR[3] = recallGPS;
//
//		//getchar();
//	}
//	return true;
//}


bool VisualLocalization::getMatchingResult(const std::vector<std::pair<double, int>>& xGPSDistance, int GPSrear, 
	const std::vector<std::pair<double, int>>& xDistance, int k, std::vector<std::pair<double, int>>& ret)
{
	vector<int> GPSidx;
	for (size_t i = 0; i < GPSrear; i++)
	{
		GPSidx.push_back(xGPSDistance[i].second);
	}
	for (size_t i = 0; i < k; i++)
	{
		// 没有相匹配的图没有这个GPS范围内的
		if (find(GPSidx.begin(), GPSidx.end(), xDistance[i].second) == GPSidx.end())
		{
			continue;
		}
		else
		{
			ret.push_back(xDistance[i]);
		}
	}	
	return true;
}

bool VisualLocalization::getMatchingResult(const std::vector<std::pair<double, int>>& xGPSDistance, int GPSrear,
	const QueryResults& ret, int k, std::vector<std::pair<double, int>>& retBoW)
{
	if (ret.empty())
	{
		return false;
	}
	vector<int> GPSidx;
	for (size_t i = 0; i < GPSrear; i++)
	{
		GPSidx.push_back(xGPSDistance[i].second);
	}
	for (size_t i = 0; i < k; i++)
	{
		// 没有相匹配的图没有这个GPS范围内的
		if (find(GPSidx.begin(), GPSidx.end(), ret[i].Id) == GPSidx.end())
		{
			continue;
		}
		else
		{
			retBoW.push_back(std::pair<double, int>(ret[i].Score, ret[i].Id));
		}
	}
	return true;
}

bool VisualLocalization::outputRet(ofstream &fResult, const std::vector<std::pair<double, int>>& xDistance, const int& bestMatchValue, 
	const int& k, const std::string& path, const int& picIdx )
{
	for (size_t i = 0; i < xDistance.size(); i++)
	{
		// 待查特征与训练集之间的距离
		double dist = -1;
		cv::Mat bestMatchedColor;
		int label;
		// 对应的训练集序号
		int idx;

		bestMatchedColor = cv::imread(descriptorbase->picsRec.getColorImgPath(xDistance[i].second));
		dist = xDistance[i].first;
		label = descriptorbase->picsRec.getLabel(xDistance[i].second);
		idx = xDistance[i].second;
		std::cout << idx << "\t";

		std::stringstream strstr;
		strstr << dist;
		cv::putText(bestMatchedColor, strstr.str(), cv::Point(50, 50), CV_FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 0, 0), 3);
		strstr.str("");

		strstr << "Label:" << label;
		strstr << " Idx:" << idx;
		cv::putText(bestMatchedColor, strstr.str(), cv::Point(50, 100), CV_FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 0, 0), 3);
		strstr.str("");
		
		strstr << path << "\\"<<picIdx<<"bestMatchedColor" <<i<<".jpg";
		cv::imwrite( strstr.str(), bestMatchedColor);
		cv::waitKey(1);

		int diff = abs(descriptorbase->distanceBase[idx] - descriptorbase->distanceBase[bestMatchValue]);
		//int diff = abs(idx - bestMatchValue);
		std::cout << diff << std::endl;
		fResult << diff << "\t";
	}
	for (size_t i = 0; i < k - xDistance.size(); i++)
	{
		fResult << -1 << "\t";
	}
	std::cout << std::endl;
	fResult << std::endl;
	return true;
}

bool VisualLocalization::outputRet(ofstream &fResult, const std::vector<std::pair<double, int>>& xDistance, const int& bestMatchValue, const std::string& path, const int& picIdx)
{
	for (size_t i = 0; i < xDistance.size(); i++)
	{		
		// 待查特征与训练集之间的距离
		double dist = -1;
		cv::Mat bestMatchedColor;		
		int label;
		// 对应的训练集序号
		int idx;

		bestMatchedColor = cv::imread(descriptorbase->picsRec.getColorImgPath(xDistance[i].second));
		dist = xDistance[i].first;
		label = descriptorbase->picsRec.getLabel(xDistance[i].second);
		idx = xDistance[i].second;
		std::cout << idx << "\t";

		std::stringstream strstr;
		strstr << dist;
		cv::putText(bestMatchedColor, strstr.str(), cv::Point(50, 50), CV_FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 0, 0), 3);
		strstr.str("");

		strstr << "Label:" << label;
		strstr << " Idx:" << idx;
		cv::putText(bestMatchedColor, strstr.str(), cv::Point(50, 100), CV_FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 0, 0), 3);
		strstr.str("");
		strstr << path << "\\" << picIdx << "bestMatchedColor" << i << ".jpg";
		cv::imwrite(strstr.str(), bestMatchedColor);
		//cv::imshow("bestMatchedColor" + strstr.str(), bestMatchedColor);
		cv::waitKey(1);

		int diff = abs(descriptorbase->distanceBase[idx] - descriptorbase->distanceBase[bestMatchValue]);
		//int diff = abs(idx - bestMatchValue);
		std::cout << diff << std::endl;
		fResult << diff << "\t";
	}

	std::cout << std::endl;
	fResult << std::endl;
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

