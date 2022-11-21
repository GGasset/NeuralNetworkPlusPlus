#include <cmath>

#pragma once
class DataManipulation
{
public:
	/// <returns>tuple<first slice, second slice, first slice length></returns>
	static tuple<float**, float**, size_t> SliceData(float** data, size_t dataLength, double slicePoint)
	{
		slicePoint = fabs(slicePoint);
		slicePoint *= 1 + (dataLength - 1) * slicePoint > 1;
		slicePoint = (double)(size_t)(slicePoint);

		float** firstSlice = new float* [slicePoint];
		float** secondSlice = new float* [dataLength - (size_t)slicePoint];

		for (size_t i = 0; i < slicePoint; i++)
		{
			firstSlice[i] = data[i];
		}

		for (size_t i = 0; i < dataLength - slicePoint; i++)
		{
			secondSlice[i] = data[(size_t)slicePoint + i];
		}

		tuple<float**, float**, size_t> output(firstSlice, secondSlice, (size_t)slicePoint);
		return output;
	}
};
