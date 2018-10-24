#include "SequenceSearch.h"

void SequenceSearch::trajectorySearch()
{
	int vmin, vmax, vstep;
	//vsteps
}

void SequenceSearch::coneSearch()//±ØÐëÎªÆæÊý
{
	if (globalResult.empty())
	{
		scoreMat = cv::Mat(matSize.height,matSize.width, CV_32FC1, 0.0f);
		return;
	}
	scoreMat.create(matSize, CV_32FC1);
	for (size_t i = 0; i < matSize.height; i++)//query
	{
		float* pS = scoreMat.ptr<float>(i);
		for (size_t j = 0; j < matSize.width; j++)//database
		{
			int count = 0;
			int min_y = std::max(0, (int)i - (numSearch - 1) / 2);
			int max_y = std::min((int)i + (numSearch - 1) / 2, matSize.height - 1);
			for (size_t k = min_y; k <= max_y; k++)//query within coneopen
			{
				int min_x, max_x;
				if (k<i)
				{
					min_x = std::max((k - i)*vmax + j, 0.0f);
					max_x =  (k - i)*vmin + j;
				}
				else
				{
					max_x = std::min(int((k - i)*vmax + j), matSize.width -1);
					min_x = (k - i)*vmin + j;
				}
				if (globalResult[k]>=min_x && globalResult[k]<=max_x)
				{
					count++;
				}
			}
			if (count)
			{
				pS[j] = float(count) / float(max_y- min_y+1);
			}
			else
			{
				pS[j] = 0;
			}
		}
	}
}

