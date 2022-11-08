#include <cmath>

#pragma once
class Cost
{
public:
	enum CostFunction
	{
		SquaredMean,
	};

	static double GetCostOf(long networkOutputLength, double* neuronOutput, double* Y, CostFunction costType)
	{
		switch (costType)
		{
		case Cost::SquaredMean:
			return SquaredMeanLoss(networkOutputLength, neuronOutput, Y);
		default:
			return NULL;
		}
	}

	static double SquaredMeanLoss(long outputLength, double* neuronOutput, double* Y)
	{
		double mean = 0;
		for (long i = 0; i < outputLength; i++)
		{
			mean += SquaredMeanLoss(neuronOutput[i], Y[i]);
		}
	}

	static double SquaredMeanLoss(double neuronOutput, double Y)
	{
		return pow(neuronOutput - Y, 2);
	}
};

