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
	int imgtype;
	fs["ImgType"] >> imgtype;
	switch (imgtype)
	{
	case 0:useColor = true; useDepth = false; useIR = false; useRGBDIR = false; useRGBIR = false; usePaperConfig = false;
		break;
	case 1:useColor = false; useDepth = true; useIR = false; useRGBDIR = false; useRGBIR = false; usePaperConfig = false;
		break;
	case 2:useColor = false; useDepth = false; useIR = true; useRGBDIR = false; useRGBIR = false; usePaperConfig = false;
		break;
	case 3:useColor = false; useDepth = false; useIR = false; useRGBDIR = true; useRGBIR = false; usePaperConfig = false;
		break;
	case 4:useColor = false; useDepth = false; useIR = false; useRGBDIR = false; useRGBIR = true; usePaperConfig = false;
		break;
	case 5:useColor = false; useDepth = false; useIR = false; useRGBDIR = false; useRGBIR = false; usePaperConfig = true;
	default:
		break;
	}
	fs["queryImgSize"] >> qImgSize;
	fs["databaseImgSize"] >> dImgSize;
	fs["Descriptor"] >> descriptor;
	fs["Color"] >> isColor;
	fs["GPS"] >> withGPS;

	return true;
}
