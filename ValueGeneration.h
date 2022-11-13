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

	static list<float> GenerateWeigths(size_t outputLength, float minValue, float valueClosestTo0, float maxValue)
	{
		list<float> output = list<float>();
		for (size_t i = 0; i < outputLength; i++)
		{
			output.push_front(GenerateWeight(minValue, valueClosestTo0, maxValue));
		}
		return output;
	}

	static tuple<list<size_t>, list<size_t>> GenerateConnectedPositions(size_t x, size_t startingY, size_t outputLength, size_t connectionsPerThread)
	{
		list<size_t> Xs, Ys;
		Xs = list<size_t>();
		Ys = list<size_t>();

		size_t nThreads = outputLength / connectionsPerThread;
		size_t remainingConnections = outputLength % connectionsPerThread;
		bool isThereARemainingThread = remainingConnections > 0;
		size_t totalThreads = nThreads + isThereARemainingThread;

		if ((nThreads + isThereARemainingThread) > 1)
		{
			thread* threads = new thread[totalThreads];
			ConnectedPositionsGenerator* positionGenerators = new ConnectedPositionsGenerator[totalThreads];

			for (size_t i = 0; i < nThreads; i++)
			{
				threads[i] = thread(std::ref(positionGenerators[i]), x, startingY + (connectionsPerThread * i), connectionsPerThread);
			}
			if (isThereARemainingThread)
				threads[nThreads] = thread(std::ref(positionGenerators[nThreads]), x, startingY + (connectionsPerThread * nThreads), remainingConnections);

			for (size_t i = 0; i < totalThreads; i++)
			{
				threads[i].join();
				DataManipulation::AddLists(&Xs, positionGenerators[i].Xs);
				DataManipulation::AddLists(&Ys, positionGenerators[i].Ys);
			}

			delete[] threads;
			delete[] positionGenerators;
		}
		else
		{
			ConnectedPositionsGenerator positionGenerator;
			thread thrd(std::ref(positionGenerator), x, startingY, outputLength);
			thrd.join();
			Xs = positionGenerator.Xs;
			Ys = positionGenerator.Ys;
		}
			
		tuple<list<size_t>, list<size_t>> output(Xs, Ys);
		return output;

	}

private:
	class ConnectedPositionsGenerator
	{
	public:
		list<size_t> Xs, Ys;

		void operator()(size_t x, size_t startingY, size_t outputLength)
		{
			Xs = list<size_t>();
			Ys = list<size_t>();

			for (size_t i = 0; i < outputLength; i++)
			{
				Xs.push_back(x);
				Ys.push_back(startingY + i);
			}
		}
	};

public:
	static float NextDouble()
	{
		return (rand() % 1000) / 1000.0F;
	}
};

