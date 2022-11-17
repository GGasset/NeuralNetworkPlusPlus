#pragma once
using namespace std;
#include <stdlib.h>
#include <math.h>
#include <thread>
#include <list>
#include <tuple>
#include "DataManipulation.h"

class ValueGeneration
{
public:
	static float GenerateWeight(float minValue, float valueClosestTo0, float maxValue)
	{
		float maxValueCp = maxValue;
		float minValueCp = minValue;
		maxValue = fmaxf(minValueCp, maxValueCp);
		minValue = fminf(minValueCp, maxValueCp);

		valueClosestTo0 = fabsf(valueClosestTo0);
		bool isMaxValuePositiveAndMinValueNegative = maxValue > 0 && minValue < 0;
		bool areMinAndMaxValuesPositive = minValue >= 0 && maxValue >= 0;
		bool areMinAndMaxValuesNegative = minValue <= 0 && maxValue <= 0;

		float output = 0;

		// output multiplier will determine if output is positive or negative only if maxValue is positive and minValue is negative
		int outputMultiplier = rand() % 2;
		outputMultiplier -= 1 * (outputMultiplier == 0);

		// set output to negative or positive valueClosestTo0 if max value is positive and min value is negative
		output += valueClosestTo0 * (isMaxValuePositiveAndMinValueNegative && (outputMultiplier == 1));
		output -= valueClosestTo0 * (isMaxValuePositiveAndMinValueNegative && (outputMultiplier == -1));

		output += minValue * areMinAndMaxValuesPositive;
		output += maxValue * areMinAndMaxValuesNegative;

		float outputAdditionMultiplier = NextDouble();

		// Set output to its final value if min value is negative and max value positive
		output += (outputAdditionMultiplier * (maxValue - valueClosestTo0)) * (isMaxValuePositiveAndMinValueNegative && outputMultiplier == 1);
		output += (outputAdditionMultiplier * (valueClosestTo0 - minValue)) * (isMaxValuePositiveAndMinValueNegative && outputMultiplier == -1);

		// Set output to its final state if both min/max values are positive or negative
		output += (outputAdditionMultiplier * (maxValue - minValue) * areMinAndMaxValuesPositive);
		output += (outputAdditionMultiplier * (minValue - maxValue) * areMinAndMaxValuesNegative);

		return output;
	}

	static float* GenerateWeigths(size_t outputLength, float minValue, float valueClosestTo0, float maxValue, size_t weightsPerThread)
	{
		size_t nThreads = outputLength / weightsPerThread;
		size_t remainingWeights = outputLength % weightsPerThread;
		bool isThereARemainingThread = remainingWeights > 0;
		size_t totalThreads = nThreads + isThereARemainingThread;

		thread* threads = new thread[totalThreads];
		WeightGenerator* weightsGenerators = new WeightGenerator[totalThreads];

		float* output = new float[outputLength];
		for (size_t i = 0; i < nThreads; i++)
		{
			threads[i] = thread(std::ref(weightsGenerators[i]), output, weightsPerThread * i, weightsPerThread, minValue, valueClosestTo0, maxValue);
		}
		if (isThereARemainingThread)
			threads[nThreads] = thread(std::ref(weightsGenerators[nThreads]), output, weightsPerThread * nThreads, remainingWeights, minValue, valueClosestTo0, maxValue);

		for (size_t i = 0; i < totalThreads; i++)
		{
			threads[i].join();
		}

		return output;
	}

private:
	class WeightGenerator
	{
	public:
		void operator()(float* weightArray, size_t startingI, size_t outputLength, float minValue, float valueClosestTo0, float maxValue)
		{
			for (size_t i = 0; i < outputLength; i++)
			{
				weightArray[startingI + i] = GenerateWeight(minValue, valueClosestTo0, maxValue);
			}
		}
	};

public:
	static tuple<size_t*, size_t*> GenerateConnectedPositions(size_t x, size_t startingY, size_t outputLength, size_t connectionsPerThread)
	{
		size_t* Xs = new size_t[outputLength];
		size_t* Ys = new size_t[outputLength];

		size_t nThreads = outputLength / connectionsPerThread;
		size_t remainingConnections = outputLength % connectionsPerThread;
		bool isThereARemainingThread = remainingConnections > 0;
		size_t totalThreads = nThreads + isThereARemainingThread;

		thread* threads = new thread[totalThreads];
		ConnectedPositionsGenerator* positionGenerators = new ConnectedPositionsGenerator[totalThreads];

		for (size_t i = 0; i < nThreads; i++)
		{
			threads[i] = thread(std::ref(positionGenerators[i]), Xs, Ys, x, startingY + (connectionsPerThread * i), connectionsPerThread);
		}
		if (isThereARemainingThread)
			threads[nThreads] = thread(std::ref(positionGenerators[nThreads]), Xs, Ys, x, startingY + (connectionsPerThread * nThreads), remainingConnections);

		for (size_t i = 0; i < totalThreads; i++)
		{
			threads[i].join();
		}

		delete[] threads;
		delete[] positionGenerators;

		tuple<size_t*, size_t*> output(Xs, Ys);
		return output;
	}

private:
	class ConnectedPositionsGenerator
	{
	public:
		void operator()(size_t* Xs, size_t* Ys, size_t x, size_t startingY, size_t outputLength)
		{
			for (size_t i = 0; i < outputLength; i++)
			{
				Xs[startingY + i] = x;
				Ys[startingY + i] = startingY + i;
			}
		}
	};

public:
	static float NextDouble()
	{
		return (rand() % 1000) / 1000.0F;
	}
};
