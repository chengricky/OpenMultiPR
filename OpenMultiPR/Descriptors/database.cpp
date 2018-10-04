#include "database.h"
#include "..\tools\Timer.h"
#include "bow/bow.h"
#include "..\tools\list_dir.h"
#include "direct.h"

Descriptorbase::Descriptorbase(GlobalConfig& config)
{
	std::vector<std::string> pathsRec;
	pathsRec.push_back(config.pathRec);


	isColor = config.isColor;

	// 对图像文件列表和GNSS、是否关键点数据进行赋值
	if (config.useColor&&config.withGPS)
	{
		picsRec.init(pathsRec, PicGnssFile::RGB, true,  "*color");
	}
	else if (config.useColor && !config.withGPS)
	{
		picsRec.init(pathsRec, PicGnssFile::RGB, false, "*");
	}
	else if(config.usePaperConfig)
	{
		picsRec.init(pathsRec, PicGnssFile::RGBDIR, true, "*color");
	}


	// load the vocabulary from disk
	DBoW3::Vocabulary voc("orbvoc.dbow3");
	ORBdb.setVocabulary(voc, false, 0); // false = do not use direct index (so ignore the last param)
										// The direct index is useful if we want to retrieve the features that belong to some vocabulary node.
										// db creates a copy of the vocabulary, we may get rid of "voc" now, add images to the database
										// loop for every images of training dataset
	int distLabel = 0;

	// set descriptor extractors
	if (config.descriptor!="Default")
	{
		if (config.descriptor=="BoW")
		{
			extraction.add(new  ORBExtractor(0));
		}
	}
	else if (config.useColor)
	{
		extraction.add(new GISTExtractor(0,true, true, config.dImgSize));
		extraction.add(new LDBExtractor(0,true));
		//extraction.add(new ORBExtractor(0));
	}
	else if (config.useDepth)
	{
		extraction.add(new GISTExtractor(1, false, false, config.dImgSize));
		extraction.add(new LDBExtractor(1, false));
		extraction.add(new ORBExtractor(1));
	}
	else if (config.useIR)
	{
		extraction.add(new GISTExtractor(2,  false, false, config.dImgSize));
		extraction.add(new LDBExtractor(2,  false));
		extraction.add(new ORBExtractor(2));
	}
	else if (config.useRGBDIR)
	{
		extraction.add(new GISTExtractor(0, true, true, config.dImgSize));
		extraction.add(new GISTExtractor(1,false, false, config.dImgSize));
		extraction.add(new GISTExtractor(2,  false, false, config.dImgSize));
		extraction.add(new LDBExtractor(0, true));
		extraction.add(new LDBExtractor(1, false));
		extraction.add(new LDBExtractor(2,  false));
		extraction.add(new ORBExtractor(0));
	}
	else if (config.useRGBIR)
	{
		extraction.add(new GISTExtractor(0,true, true, config.dImgSize));
		extraction.add(new GISTExtractor(2, false, false, config.dImgSize));
		extraction.add(new LDBExtractor(0,true));
		extraction.add(new LDBExtractor(2, false));
		extraction.add(new ORBExtractor(0));
	}
	else if (config.usePaperConfig)
	{
		extraction.add(new GISTExtractor(0, true, true, config.dImgSize));
		extraction.add(new LDBExtractor(0, true));
		extraction.add(new LDBExtractor(1, false));
		extraction.add(new LDBExtractor(2,  false));
		//extraction.add(new ORBExtractor(0));
	}

	while (picsRec.doMain())
	{
		/*GPS*/
		xGPSRec.push_back(std::pair<double, double>(picsRec.longitudeValue, picsRec.latitudeValue));

		// Mat order color-depth-IR
		std::vector<cv::Mat> todoImages = getAllImage(picsRec, config.dImgSize);
		std::vector<float> xGIST;
		cv::Mat xLDB;//height1 width173		
		cv::Mat xORB;//height32 width...(distance相加？)//由于只有一个

		// save the descriptor of GIST and LDB
		Timer timer;
		timer.start();

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
				_mkdir("ORB_DB");
				strstream << "orb" << picsRec.getFilePointer() << ".jpg";
				cv::imwrite("ORB_DB\\"+strstream.str(), mat);
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
		
		timer.stop();
		std::cout << "REF: ";
		timer.print_elapsed_time(TimeExt::MSec);

		/*compressive sensing*/
		//arma::Col<klab::DoubleReal> xCS;
		//Mat grayCS;
		//if (isColor)
		//{
		//	cvtColor(todoImages[0], grayCS, CV_BGR2GRAY);
		//}
		//else
		//{
		//	grayCS = todoImages[0];
		//}
		//if(extractCS(grayCS, xCS))
		//	xCSRec.push_back(xCS);	

		int num = xGIST.size();
		/*GIST*/
		xGISTRec.push_back(xGIST);

		/*LDB*/
		xLDBRec.push_back(xLDB);		

		/*BoW*/
		DBoW3::BowVector bowvec;
		DBoW3::FeatureVector fvec;
		ORBdb.add(xORB, &bowvec, &fvec);
		//std::cout << "//////////////////////////////////" << std::endl;
		//std::cout << bowvec.size() << std::endl;
		//std::cout << fvec.size() << std::endl;

		/*ORB*/
		xORBRec.push_back(xORB);

		////////// 论文显示使用
		//BowVector::iterator it_bowvec = bowvec.begin();
		//while (it_bowvec != bowvec.end())
		//{
		//	std::cout << it_bowvec->first << "\t";
		//	std::cout<<it_bowvec->second<<std::endl;
		//	it_bowvec++;
		//}
		//FeatureVector::iterator it_fvec = fvec.begin();
		//while (it_fvec != fvec.end())
		//{
		//	std::cout << it_fvec->first << "\t";
		//	std::cout << it_fvec->second << std::endl;
		//	it_fvec++;
		//}
		//imshow("orig", todoImages[0]);
		//imwrite("orig.png", todoImages[0]);
		//Mat kpMat = todoImages[0];
		//for each (auto var in keypoints)
		//{
		//	circle(kpMat, var.pt, var.size/20, Scalar(255, 0, 0));
		//	//std::cout << var.angle;
		//}
		//std::cout << keypoints.size();
		//imshow("key", kpMat);
		//imwrite("key.png", kpMat);
		//waitKey(1);
		//// 转换成按位的
		//Mat redHist(xORB.rows, xORB.cols * 8, CV_8UC1, Scalar(0));
		//for (size_t j = 0; j < redHist.rows; j++)
		//{
		//	for (size_t i = 0; i < redHist.cols; i++)
		//	{
		//		redHist.at<uchar>(j, i) = bool((xORB.at<uchar>(j, i / 8))&uchar(pow(2, 7 - i % 8))) * 255;
		//	}
		//}
		//int scale = 1;
		//int histHeight = 128;
		//int bins = 32;
		//Mat histImage(histHeight, redHist.cols, CV_8UC3,Scalar(255,255,255));
		//double maxValue_red;
		//minMaxLoc(redHist.row(0), 0, &maxValue_red, 0, 0);
		////正式开始绘制
		//for (int i = 0; i<bins*8; i++)
		//{
		//	//参数准备
		//	float binValue_red = redHist.at<uchar>(i);
		//	int intensity_red = cvRound(binValue_red*histHeight / maxValue_red);  //要绘制的高度
		//																			 //绘制红色分量的直方图
		//	line(histImage, Point(i, histHeight - 1), Point(i, histHeight - intensity_red), CV_RGB(0, 0, 255));
		//}
		////std::cout << redHist << std::endl;
		//imshow("ALL ORB", ~redHist);
		//imwrite("ALLORB.png", ~redHist);
		//imshow("ORB", histImage);
		//imwrite("ORB.png", histImage);
		//waitKey(1);
		//waitKey(0);
		////////////////////

		/*distance base*/
		if (!picsRec.posLabelValue)
		{
			distLabel++;
		}
		distanceBase.push_back(distLabel);
		isLabel.push_back(picsRec.posLabelValue);
		cv::waitKey(1);
	}
	//std::cout << "Database information: " << std::endl << ORBdb << std::endl;
	//assert(!xLDBRec.empty());

}


std::vector<cv::Mat> Descriptorbase::getAllImage(const PicGnssFile& picsRec, const cv::Size& imgSize )
{
	// Mat order color-depth-IR
	std::vector<cv::Mat> todoImages(3);


	cv::resize(picsRec.colorImg, todoImages[0], imgSize);
	if (!isColor)
	{
		cv::Mat tmp;
		cvtColor(picsRec.colorImg, tmp, CV_BGR2GRAY);
		cv::resize(tmp, todoImages[0], imgSize);
	}
	if (!picsRec.depthImg.empty())
	{
		cv::resize(picsRec.depthImg, todoImages[1], imgSize);
	}
	if (!picsRec.IRImg.empty())
	{
		cv::resize(picsRec.IRImg, todoImages[2], imgSize);
	}


	return todoImages;
}

