#pragma once
#include "Header.h"

class SequenceSearch
{
	cv::Mat distanceMat;
public:
	SequenceSearch(cv::Mat distanceMat) : distanceMat(distanceMat){};
	~SequenceSearch() {};

	void trajectorySearch();
	void coneSearch(int numSearch, float vmin, float vmax, bool isMax);

	void globalSearch(bool isMax);

	cv::Mat scoreMat;
	std::vector<int> globalResult;//0-based
};