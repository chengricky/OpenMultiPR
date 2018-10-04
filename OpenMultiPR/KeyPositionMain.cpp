#include "FileInterface\picGnssfile.h"
#include "descriptors/CS/CSOperation.h"
#include "VisualLocalization.h"
#include "FileInterface\GlobalConfig.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <opencv2/opencv.hpp>

struct Params
{
	int kLDB;
	int kGIST;
	int kBoW;
	double r;
	double numKeyRatio;
	double numKeyTh;
	double PandR[4];
};

GlobalConfig GlobalConfig::config("Config.yaml");

int main()
{
	static GlobalConfig& config = GlobalConfig::instance();

	// object for visual localization
	//try
	//{
	VisualLocalization vl(config);
	double PandR[4];
	//bool VisualLocalization::test(const bool& outIdxDev, const bool &outPR, double* PandR, int kLDB, int kGIST, int kBoW, double r, double numKeyRatio, double numKeyTh)
	if (config.usePaperConfig)
	{
		vl.paperLocalize(false, PandR, true, 7, 9, 7, 0.01, 0.5, 2);
	}
	else if (config.withGPS)
	{
		vl.GPSLocalize();
	}
	else
	{
		vl.test(config, true, false, PandR, 7, 9, 7, 0.01, 0.5, 2);
	}

	//}
	//catch (const std::exception&)
	//{
	//	std::cout << "exception";
	//	system("pause");
	//	return -1;
	//}

	//vl.preparePara();
	//// match descriptor with recorded images
	//vector<int> k = { 1,3,5,7,9 };
	//vector<double> r = { 0.005,0.01,0.015,0.02,0.025,0.03,0.035 };
	//vector<double> ratio = { 0.3,0.4,0.5,0.6,0.7 };
	//vector<double> th = { 2,3,4,5 };
	//Params* paras = new Params[k.size()*k.size()*k.size()*r.size()*ratio.size()*th.size()];
	//int idx = 0;
	//for each (auto varkLDB in k)
	//{
	//	for each(auto varkGIST in k)
	//	{
	//		for each (auto varkBoW in k)
	//		{
	//			for each(auto varr in r)
	//			{
	//				for each(auto varratio in ratio)
	//				{
	//					for each(auto varth in th)
	//					{
	//						paras[idx].kBoW = varkBoW;
	//						paras[idx].kGIST = varkGIST;
	//						paras[idx].kLDB = varkLDB;
	//						paras[idx].numKeyRatio = varratio;
	//						paras[idx].numKeyTh = varth;
	//						paras[idx].r = varr;
	//						idx++;
	//					}
	//				}
	//			}
	//		}
	//	}
	//}
	//for (size_t i = 0; i < k.size()*k.size()*k.size()*r.size()*ratio.size()*th.size(); i++)
	//{
	//	state = vl.testPara(false, true, paras[i].PandR, paras[i].kLDB, paras[i].kGIST, paras[i].kBoW, paras[i].r, paras[i].numKeyRatio, paras[i].numKeyTh);
	//	if (!state)
	//	{
	//		std::cout << "Test failed!" << std::endl;
	//	}
	//	
	//}
	//ofstream fcsv("PandR.csv");
	//for (size_t i = 0; i < k.size()*k.size()*k.size()*r.size()*ratio.size()*th.size(); i++)
	//{
	//	fcsv << paras[i].kLDB << "," << paras[i].kGIST << "," << paras[i].kBoW << "," << paras[i].r << "," << paras[i].numKeyRatio << "," << paras[i].numKeyTh
	//		<< "," << paras[i].PandR[0] << "," << paras[i].PandR[1] << "," << paras[i].PandR[2] << "," << paras[i].PandR[3]<<std::endl;
	//}

	system("pause");
	return 0;
}
