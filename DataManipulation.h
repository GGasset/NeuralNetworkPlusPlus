#include <cmath>
#include <tuple>

#pragma once
static class DataManipulation
{
public:
	static void ShuffleData(float** data, size_t dataLength)
	{
		for (size_t i = 0; i < dataLength; i++)
		{
			int randomI = rand() % (dataLength + 1);

			float* a = data[i];
			float* b = data[randomI];

			data[randomI] = a;
			data[i] = b;
		}
	}

	static size_t GetSlicePoint(size_t dataLength, double slicePoint)
	{
		slicePoint = fabs(slicePoint);
		slicePoint *= 1 + (dataLength - 1) * (slicePoint > 1);
		slicePoint = (double)(size_t)(slicePoint);
		return (size_t)slicePoint;
	}
};
