#include "GlobalConfig.h"



GlobalConfig::GlobalConfig(std::string ymlPath) 
{
	fs.open(ymlPath, cv::FileStorage::READ);
	valid = true;
	if (!readConfig())
	{
		valid = false;
	}
}

bool GlobalConfig::readConfig()
{
	if (!fs.isOpened())
	{
		return false;
	}
	fs["PathRec"] >> pathRec;
	fs["PathTest"] >> pathTest;
	fs["BoW_CodeBook"] >> codeBook;
	fs["ColorImg"] >> useColor;
	fs["DepthImg"] >> useDepth;
	fs["InfraredImg"] >> useIR;
	fs["queryImgSize"] >> qImgSize;
	fs["databaseImgSize"] >> dImgSize;
	fs["GIST"] >> useGIST;
	fs["ORB-BoW"] >> useBoW;
	fs["LDB"] >> useLDB;
	fs["Color"] >> isColor;
	fs["GPS"] >> withGPS;
	fs["Mode"] >> mode;

	return true;
}
