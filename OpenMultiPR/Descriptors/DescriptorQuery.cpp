#include "DescriptorQuery.h"
#include "..\tools\list_dir.h"
#include <direct.h>

using namespace std;
using namespace cv;

DescriptorQuery::DescriptorQuery(GlobalConfig& config)
{
	pathsTest.push_back(config.pathTest);

	if (config.useColor&&!config.withGPS)
	{
		picsTest = new PicGnssFile(pathsTest, PicGnssFile::RGB, false, 1,"*color");
	}
	else if (config.useColor && config.withGPS)
	{
		picsTest = new PicGnssFile(pathsTest, PicGnssFile::RGB, true, 1, "*color");
	}
	else if (config.usePaperConfig)
	{
		picsTest = new PicGnssFile(pathsTest, PicGnssFile::RGBDIR, true, 1,"*color");
	}

	isColor = config.isColor;
	// set descriptor extractors
	if (config.descriptor != "Default")
	{
		if (config.descriptor == "BoW")
		{
			extraction.add(new  ORBExtractor(0));
		}
	}
	else if (config.useColor)
	{
		extraction.add(new GISTExtractor(0, true, true, config.qImgSize));
		extraction.add(new LDBExtractor(0, true));
		//extraction.add(new ORBExtractor(0));
	}
	else if (config.useDepth)
	{
		extraction.add(new GISTExtractor(1, false, false, config.qImgSize));
		extraction.add(new LDBExtractor(1, false));
		extraction.add(new ORBExtractor(1));
	}
	else if (config.useIR)
	{
		extraction.add(new GISTExtractor(2, false, false, config.qImgSize));
		extraction.add(new LDBExtractor(2, false));
		extraction.add(new ORBExtractor(2));
	}
	else if (config.useRGBDIR)
	{
		extraction.add(new GISTExtractor(0, true, true, config.qImgSize));
		extraction.add(new GISTExtractor(1, false, false, config.qImgSize));
		extraction.add(new GISTExtractor(2, false, false, config.qImgSize));
		extraction.add(new LDBExtractor(0, true));
		extraction.add(new LDBExtractor(1, false));
		extraction.add(new LDBExtractor(2, false));
		extraction.add(new ORBExtractor(0));
	}
	else if (config.useRGBIR)
	{
		extraction.add(new GISTExtractor(0, true, true, config.qImgSize));
		extraction.add(new GISTExtractor(2, false, false, config.qImgSize));
		extraction.add(new LDBExtractor(0, true));
		extraction.add(new LDBExtractor(2, false));
		extraction.add(new ORBExtractor(0));
	}
	else if (config.usePaperConfig)
	{
		extraction.add(new GISTExtractor(0, true, true, config.qImgSize));
		extraction.add(new LDBExtractor(0, true));
		extraction.add(new LDBExtractor(1, false));
		extraction.add(new LDBExtractor(2, false));
		//extraction.add(new ORBExtractor(0));
	}
	while (picsTest->doMain())
	{
		/*GPS*/
		xGPSQuery.push_back(std::pair<double, double>(picsTest->longitudeValue, picsTest->latitudeValue));

		/*compressive sensing*/
		arma::Col<klab::DoubleReal> xCS;
		/*GIST*/
		std::vector<float> xGIST;
		/*LDB*/
		cv::Mat xLDB;
		/*BoW*/
		cv::Mat xORB;
		//QueryResults ret;//BoW直接得到kNN结果ret
		/*SURF*/
		cv::Mat descriptorsSURF;
		cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("FlannBased");
		extractDescriptor(config, xCS, xGIST, xLDB, xORB, descriptorsSURF);
		
		xGISTQuery.push_back(xGIST);
		xLDBQuery.push_back(xLDB);
		xORBQuery.push_back(xORB);
	}
	
	std::cout << "DescriptorQuery is built!" << std::endl;
}

DescriptorQuery::~DescriptorQuery()
{
	if (picsTest!=nullptr)
	{
		delete picsTest;
	}
}

std::vector<cv::Mat> DescriptorQuery::getAllImage(const cv::Size& imgSize)
{
	// Mat order color-depth-IR
	std::vector<cv::Mat> todoImages(3);

	cv::resize(picsTest->colorImg, todoImages[0], imgSize);
	if (!isColor)
	{
		cv::Mat tmp; 
		cvtColor(picsTest->colorImg, tmp, CV_BGR2GRAY);
		cv::resize(tmp, todoImages[0], imgSize);
	}
	if (!picsTest->depthImg.empty())
	{
		cv::resize(picsTest->depthImg, todoImages[1], imgSize);
	}
	if (!picsTest->IRImg.empty())
	{
		cv::resize(picsTest->IRImg, todoImages[2], imgSize);
	}

	return todoImages;
}

bool DescriptorQuery::extractDescriptor(GlobalConfig& config, arma::Col<klab::DoubleReal>& xCS, std::vector<float>& xGIST, cv::Mat& xLDB, cv::Mat& xORB, cv::Mat& descriptorsSURF)
{
	std::vector<cv::Mat> todoImages = getAllImage(config.qImgSize);

	cv::Mat showImg = todoImages[0];
	//std::stringstream strs;
	//strs << "best:" << picsTest->bestMatchValue;
	//cv::putText(showImg, strs.str(), cv::Point(50, 50), CV_FONT_HERSHEY_PLAIN, 2, cv::Scalar(255), 3);
	//cv::imshow("todoImage", showImg);
	cv::waitKey(1);

	/*compressive sensing*/
	//if (todoImages[0].channels()==1)
	//{
	//	extractCS(todoImages[0], xCS);
	//}
	//else
	//{
	//	Mat grayCS;
	//	cvtColor(todoImages[0], grayCS, CV_BGR2GRAY);
	//	extractCS(grayCS, xCS);
	//}

	extraction.run(todoImages);
	std::vector<cv::Mat> xLDBChannel;
	std::vector<std::vector<float>> xGISTChannel;
	for (size_t i = 0; i < extraction.getSize(); i++)
	{
		ImgDescriptorExtractor* pDescriptor = extraction.getResult(i);
		if (typeid(*pDescriptor) == typeid(GISTExtractor))
		{
			GISTExtractor* pGIST = static_cast<GISTExtractor*>(pDescriptor);
			xGISTChannel.push_back(pGIST->getResult());
		}
		else if (typeid(*pDescriptor) == typeid(LDBExtractor))
		{
			LDBExtractor* pLDB = static_cast<LDBExtractor*>(pDescriptor);
			xLDBChannel.push_back(pLDB->getResult());
		}
		else if(typeid(*pDescriptor)==typeid(ORBExtractor))
		{
			ORBExtractor* pORB = static_cast<ORBExtractor*>(pDescriptor);
			(pORB->getResult()).copyTo(xORB);
			cv::Mat mat;
			cv::drawKeypoints(todoImages[0], pORB->getKeypoints(), mat);
			cv::imshow("orb", mat);
			std::stringstream strstream;
			_mkdir("ORB_QR");
			strstream << "orb" << picsTest->getFilePointer() << ".jpg";
			cv::imwrite("ORB_QR\\" + strstream.str(), mat);
		}
	}
	if (!xLDBChannel.empty())
	{
		cv::merge(xLDBChannel, xLDB);
	}

	for (size_t i = 0; i < xGISTChannel.size(); i++)
	{
		xGIST.insert(xGIST.end(), xGISTChannel[i].begin(), xGISTChannel[i].end());
	}


	/*BoW*/
	//// ORB descriptor extraction
	//cv::Ptr<cv::ORB> orb = cv::ORB::create();
	//vector<cv::KeyPoint> keypoints;
	//cv::Mat descriptors;
	//orb->detectAndCompute(todoImages[0], cv::Mat(), keypoints, descriptors);
	//ORBdb.query(xORB, ret,-1);
	//// ret[0] is always the same image in this case, because we added it to the 
	//// database. ret[1] is the second best match.
	////std::cout << "Searching for Image " << ret << std::endl;

	/*SURF*/
	/*cv::Ptr<cv::FeatureDetector> surfdetector = cv::xfeatures2d::SURF::create();
	cv::Ptr<cv::DescriptorExtractor> surfextractor = cv::xfeatures2d::SURF::create();

	std::vector<cv::KeyPoint> keypointsSURF;
	surfdetector->detect(todoImages[0], keypointsSURF);
	surfextractor->compute(todoImages[0], keypointsSURF, descriptorsSURF);*/

	return true;
}


void DescriptorQuery::loadFeatureFromFile(const std::string &filename, std::vector<unsigned char>& dim) {
	// Timer timer;
	// timer.start();
	std::ifstream in(filename.c_str());
	if (!in) {
		printf("[ERROR][Descriptorbase] Feature %s cannot be loaded\n",
			filename.c_str());
		getchar();
		exit(EXIT_FAILURE);
	}
	else
	{
		std::cout << "Reading Query: " << filename << " ..." << std::endl;
	}
	//std::string path;
	//in >> path;

	while (!in.eof()) {
		unsigned short value;
		in >> value;
		dim.push_back(value);
	}
	in.close();
	dim.shrink_to_fit();
	// timer.stop();
	// cout << "Feature loading time: ";
	// timer.print_elapsed_time(TimeExt::MSec);
	//binarize();
}