#include "DescriptorExtraction.h"
#include "CS/CSOperation.h"
#include "GIST\include\gist.h"
#include <opencv2\xfeatures2d.hpp>


ORBExtractor::ORBExtractor(int imgIdx): ImgDescriptorExtractor(imgIdx)
{
	// BoW使用默认配置获得ORB特征
	orb = cv::ORB::create(500, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
}

bool ORBExtractor::extract(std::vector<cv::Mat> todoImages)
{
	std::vector<cv::KeyPoint> keypoint;

	orb->detectAndCompute(todoImages[imgIdx], cv::Mat(), keypoints, result);//描述子数目与关键点数相同，现在是每个描述子是32维,256维度的二进制数值
																	 // add images to the database
	return true;
}

//SURFExtractor::SURFExtractor(cv::Mat const&  img) : ImgDescriptorExtractor(img)
//{
//	surfdetector = cv::xfeatures2d::SURF::create();
//	cv::Ptr<cv::ImgDescriptorExtractor> surfextractor = cv::xfeatures2d::SURF::create();
//
//}
//
//bool SURFExtractor::extract()
//{
//	surf->detectAndCompute(todoImage, cv::Mat(), keypoints, result);//描述子数目与关键点数相同，现在是每个描述子是32维,256维度的二进制数值
//																   // add images to the database
//}


GISTExtractor::GISTExtractor(int imgIdx,  bool useColor, bool isNormalize, cv::Size imgSize): ImgDescriptorExtractor(imgIdx),
	GIST_PARAMS({ useColor, imgSize.width, imgSize.height, 4, 3, { 8, 8, 8 } })
{

}

bool GISTExtractor::extract(std::vector<cv::Mat> todoImages)
{
	cls::GIST gist_ext(GIST_PARAMS);
	gist_ext.extract(todoImages[imgIdx], result, isNormalize);//输入彩色图，内部会自动根据DEFAULT_PARAMS需要，转换灰度

	return true;
}

CSExtractor::CSExtractor(int imgIdx, cv::Size imgSize) : ImgDescriptorExtractor(imgIdx), imgSize(imgSize)
{

}

bool CSExtractor::extract(std::vector<cv::Mat> todoImages)
{
	if (todoImages[imgIdx].channels() != 1)
	{
		std::cout << "Compressive sensing does not support color image!" << std::endl;
		return false;
	}
	try
	{
		cv::Size imgSizeCS = cv::Size(imgSize.width / 2, imgSize.height / 2);
		CSCreater csCreater(imgSizeCS, 0.1, true, false);
		result = csCreater.generateCS(todoImages[imgIdx]);
	}
	catch (klab::KException& e)
	{
		std::cout << "ERROR! KLab exception : " << klab::FormatExceptionToString(e) << std::endl;
		return false;
	}
	catch (std::exception& e)
	{
		std::cout << "ERROR! Standard exception : " << klab::FormatExceptionToString(e) << std::endl;
		return false;
	}
	catch (...)
	{
		std::cout << "ERROR! Unknown exception !" << std::endl;
		return false;
	}
	return true;
}

LDBExtractor::LDBExtractor(int imgIdx, bool useColor) : ImgDescriptorExtractor(imgIdx), useColor(useColor)
{

}

bool LDBExtractor::extract(std::vector<cv::Mat> todoImages)
{
	// Select the central keypoint
	std::vector<cv::KeyPoint> kpts;
	cv::KeyPoint kpt;
	kpt.pt.x = todoImages[imgIdx].cols / 2 + 1;
	kpt.pt.y = todoImages[imgIdx].rows / 2 + 1;
	kpt.size = 1.0;
	kpt.angle = 0.0;
	kpts.push_back(kpt);
	LDB ldb;
	// attention: color image are not supported in the ldb extraction 内部会转换为灰度图
	cv::Mat matrix;
	if (todoImages[imgIdx].channels() != 1 && useColor)
		//cvtColor(illumination_conversion(todoImage), matrix, cv::COLOR_BGR2GRAY);
		matrix = illumination_conversion(todoImages[imgIdx]);
	else
		matrix = todoImages[imgIdx];
	ldb.compute(matrix, kpts, result);

	return true;
}

/**
* @brief Converts an image to illumination invariant image.
* @param image Computed image
* @return Illumination invariant image
*/
cv::Mat LDBExtractor::illumination_conversion(cv::Mat image) {

	float alpha = 0.47;

	std::vector<cv::Mat> channels(3);
	split(image, channels);

	cv::Mat imageB, imageG, imageR;
	cv::Mat imageI = cv::Mat(cv::Size(image.cols, image.rows), CV_32FC1);
	cv::Mat imageI8U = cv::Mat(cv::Size(image.cols, image.rows), CV_8UC1);

	channels[0].convertTo(imageB, CV_32FC1, 1.0 / 255.0, 0);
	channels[1].convertTo(imageG, CV_32FC1, 1.0 / 255.0, 0);
	channels[2].convertTo(imageR, CV_32FC1, 1.0 / 255.0, 0);

	float valueG, valueB, valueR;

	for (int i = 0; i < imageI.rows; i++) {

		for (int j = 0; j < imageI.cols; j++) {

			if (imageG.at<float>(i, j) != 0)
				valueG = log(imageG.at<float>(i, j));
			else
				valueG = 0;

			if (imageB.at<float>(i, j) != 0)
				valueB = alpha*log(imageB.at<float>(i, j));
			else
				valueB = 0;

			if (imageR.at<float>(i, j) != 0)
				valueR = (1 - alpha)*log(imageR.at<float>(i, j));
			else
				valueR = 0;

			imageI.at<float>(i, j) = 0.5 + valueG - valueB - valueR;

			if (imageI.at<float>(i, j)<0) imageI.at<float>(i, j) = 1;

		}

	}

	imageI.convertTo(imageI8U, CV_8UC1, 255.0, 0);

	return imageI8U;

}

ImgDescriptorExtractor* Extraction::getResult(int idx)
{
	if (idx>=extractions.size())
	{
		throw std::range_error("exceed the number of extractor.");
	}
	return extractions[idx];
}

Extraction::~Extraction()
{
	for (size_t i = 0; i < extractions.size(); i++)
	{
		delete extractions[i];
	}
	extractions.clear();
}

void Extraction::run(std::vector<cv::Mat> const&  todoImages) {

	auto it = extractions.begin(); 

	while (it != extractions.end()) 
		(*it++)->extract(todoImages);
}