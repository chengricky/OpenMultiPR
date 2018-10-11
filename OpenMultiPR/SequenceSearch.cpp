#include "SequenceSearch.h"

void SequenceSearch::trajectorySearch()
{
	int vmin, vmax, vstep;
	//vsteps
}

void SequenceSearch::coneSearch(int numSearch, float vmin, float vmax, bool isMax)//±ØÐëÎªÆæÊý
{
	if (globalResult.empty())
	{
		globalSearch(isMax);
	}
	scoreMat.create(distanceMat.size(), distanceMat.type());
	for (size_t i = 0; i < distanceMat.rows; i++)//query
	{
		float* pD = distanceMat.ptr<float>(i);
		float* pS = scoreMat.ptr<float>(i);
		for (size_t j = 0; j < distanceMat.cols; j++)//database
		{
			int count = 0;
			int min_y = std::max(0, (int)i - (numSearch - 1) / 2);
			int max_y = std::min((int)i + (numSearch - 1) / 2, distanceMat.cols - 1);
			for (size_t k = min_y; k <= max_y; k++)//query within cone
			{
				int min_x, max_x;
				if (k<i)
				{
					min_x = std::max((k - i)*vmax + j, 0.0f);
					max_x =  (k - i)*vmin + j;
				}
				else
				{
					max_x = std::min(int((k - i)*vmax + j), distanceMat.cols-1);
					min_x = (k - i)*vmin + j;
				}
				if (globalResult[k]>=min_x && globalResult[k]<=max_x)
				{
					count++;
				}

			}
			if (count)
			{
				pS[j] = (float)count / numSearch;
			}
			else
			{
				pS[j] = 0;
			}
		}
	}
}

void SequenceSearch::globalSearch(bool isMax)
{
	for (size_t i = 0; i < distanceMat.rows; i++)//query
	{
		double minVal, maxVal;
		int minPos, maxPos;
		cv::minMaxIdx(distanceMat.row(i), &minVal, &maxVal, &minPos, &maxPos);
		if (isMax)
		{
			globalResult.push_back(maxPos);
		}
		else
		{
			globalResult.push_back(minPos);
		}
		
	}
}