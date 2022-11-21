// NeuralNetwork++.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

using namespace std;
#include <iostream>
#include "NeuralNetwork.h"

int main()
{
	size_t shape[] = { 1, 150, 100, 5, 1};
	NeuralNetwork n = NeuralNetwork(5, shape, false,ActivationFunctions::Sigmoid, 1, -.5, 0, .5);

	float X[] = {3.5f};
	
	float Y[] = {.987654321f};

	float* output = n.Execute(X);
	std::cout << output[0] << "\n";
	delete[] output;

	for (size_t i = 0; i < 5000; i++)
	{
		n.ApplyGradients(n.GetGradients(X, Y, Cost::SquaredMean), .75);
		output = n.Execute(X);
		std::cout << output[0] << "\n";

		delete[] output;
	}

	n.Dispose();
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
