#pragma once
#include <stdlib.h>
#include <math.h>

static class ValueGeneration
{

public:
	static double GenerateWeight(double maxValue, double minValue, double valueClosestTo0)
	{
		double maxValueCp = maxValue;
		double minValueCp = minValue;
		maxValue = fmax(minValueCp, maxValueCp);
		minValue = fmin(minValueCp, maxValueCp);

		valueClosestTo0 = fabs(valueClosestTo0);
		bool isMaxValuePositiveAndMinValueNegative = maxValue > 0 && minValue < 0;
		bool areMinAndMaxValuesPositive = minValue >= 0 && maxValue >= 0;
		bool areMinAndMaxValuesNegative = minValue <= 0 && maxValue <= 0;

		double output = 0;

		// output multiplier will determine if output is positive or negative only if maxValue is positive and minValue is negative
		int outputMultiplier = rand() % 2;
		outputMultiplier -= 1 * (outputMultiplier == 0);

		// set output to negative or positive valueClosestTo0 if max value is positive and min value is negative
		output += valueClosestTo0 * (isMaxValuePositiveAndMinValueNegative && (outputMultiplier == 1));
		output -= valueClosestTo0 * (isMaxValuePositiveAndMinValueNegative && (outputMultiplier == -1));

		output += minValue * areMinAndMaxValuesPositive;
		output += maxValue * areMinAndMaxValuesNegative;

	}
};

