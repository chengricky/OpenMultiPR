#pragma once
#include "..\Header.h"

class GroundTruth
{
	std::vector<int> groundTruth;
	std::vector<bool> keyLabel;

public:
	GroundTruth(std::string filepathGT, std::string filepathLabel);
	~GroundTruth() {};
	
	std::vector<std::vector<int>> gt;
	void generateGroundTruth(int Tolerence);
};
