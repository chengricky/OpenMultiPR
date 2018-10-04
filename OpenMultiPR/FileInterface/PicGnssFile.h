#pragma once
#include "FileInterface.h"
#include <fstream>
#include <string>

class PicGnssFile : public FileInterface
{
public:
	PicGnssFile() {}
	PicGnssFile(std::vector<std::string> folderPath, int mode, bool ifLabeledCW, int sequenceNum, std::string fileKeyWord);
	void init(std::vector<std::string> folderPath, int mode, bool ifLabeledCW, std::string fileKeyWord);
	void init(std::vector<std::string> folderPath, int mode, bool ifLabeledCW, int sequenceNum, std::string fileKeyWord);
	void init(std::vector<std::string> folderPath, int mode, bool ifLabeledCW) {};
	bool doMain();
	cv::Size getImgSize();

	//std::vector<cv::Mat> colorImg;
	//std::vector<cv::Mat> depthImg;
	//std::vector<cv::Mat> IRImg;
	cv::Mat colorImg;
	cv::Mat depthImg;
	cv::Mat IRImg;
	double latitudeValue;
	double longitudeValue;
	int posLabelValue;	//�Ƿ���·���ϵĹؼ���
	int bestMatchValue;
	bool readVideo() { return false; }
	std::string getColorImgPath(int imgIdx){ return colorFiles[imgIdx]; };
	int getLabel(int imgIdx){ return posLabel[imgIdx]; };
	unsigned long getFilePointer(){ return filePointer;};
	unsigned long getFileVolume(){ return fileVolume; };
	int getBestMatch(int i) { return bestMatch[i]; };
	int getPosLabel(int i) { return posLabel[i]; };

	const static int RGB = 1;
	const static int RGBD = 2;
	const static int RGBDIR = 3;//��IRͼ��

private:
	int mode;
	bool ifLabelCW;
	std::vector<std::string> depthFiles;
	std::vector<std::string> colorFiles;
	std::vector<std::string> IRFiles;
	std::vector<std::string> labelFiles;
	std::vector<double> latitude; //γ��
	std::vector<double> longitude; //����
	std::vector<int> posLabel;
	std::vector<int> bestMatch;
	unsigned long filePointer;
	unsigned long fileVolume;
	int sequenceNum;

	void findFilesfromColor(std::string path, std::string prefix, std::string suffix, bool& depth, bool&ground, bool& label);

	// ���ڶ�ȡ������txt����ļ�
	std::ifstream txtGnssLabel;
	bool readTxtGnssLabel();

};