using namespace std;
#include "Neuron.h"
#include <list>
#include <tuple>

#pragma once
class NeuralNetwork
{
private:
	list<list<Neuron>> Neurons;
	ActivationFunctions::ActivationFunction ActivationFunction;
	int OutputLength;

public:
	NeuralNetwork(long shapeLength, long shape[], bool deleteShapeArr, ActivationFunctions::ActivationFunction activationFunction, double bias, double minWeight, double weightClosestTo0, double maxWeight)
	{
		Neurons = list<list<Neuron>>();
		for (long i = 1; i < shapeLength; i++)
		{
			list<Neuron> currentLayer = list<Neuron>();
			for (long j = 0; j < shape[i]; j++)
			{
				currentLayer.push_back(Neuron(i, shape[i - 1], bias, minWeight, weightClosestTo0, maxWeight));
			}
			Neurons.push_back(currentLayer);
		}

		OutputLength = shape[shapeLength - 1];

		ActivationFunction = activationFunction;
		if (deleteShapeArr)
			delete[] shape;
	}

private:
	NeuralNetwork(list<list<Neuron>> neurons, ActivationFunctions::ActivationFunction activationFunction, int outputLength)
	{
		Neurons = neurons;
		ActivationFunction = activationFunction;
		OutputLength = outputLength;
	}

public:
	double* Execute(double* input)
	{
		tuple<double**, double**> storedExecution = ExecuteStore(input);
		double** linears = get<0>(storedExecution);
		double** activations = get<1>(storedExecution);

		double* output = activations[GetNetworkLayerCount()];

		delete[] linears[0];
		for (size_t i = 1; i < GetNetworkLayerCount(); i++)
		{
			delete[] linears[i];
			delete[] activations[i];
		}
		delete[] linears;
		delete[] activations;

		return output;
	}

private:
	/// <summary>
	/// 
	/// </summary>
	/// <param name="input"></param>
	/// <returns>tuple(networkLinears, networkActivations)</returns>
	tuple<double**, double**> ExecuteStore(double* input)
	{
		double** networkLinearFunctions = new double* [GetNetworkLayerCount()];
		double** networkActivations = new double* [GetNetworkLayerCount() + 1];
		networkActivations[0] = input;
		auto layerIter = Neurons.begin();
		for (long i = 0; i < GetNetworkLayerCount(); i++, layerIter++)
		{
			tuple<double*, double*> layerExecutionResults = ExecuteStoreLayer(layerIter, networkActivations);
			networkLinearFunctions[i] = get<0>(layerExecutionResults);
			networkActivations[i + 1] = get<1>(layerExecutionResults);
		}

		tuple<double**, double**> output(networkLinearFunctions, networkActivations);
		return output;
	}

	/// <summary>
	/// 
	/// </summary>
	/// <param name="layerIter"></param>
	/// <param name="networkActivations"></param>
	/// <returns>tuple(linearFunctions, networkActivations)</returns>
	tuple<double*, double*> ExecuteStoreLayer(list<list<Neuron>>::iterator layerIter, double** networkActivations)
	{
		list<Neuron> layer = (*layerIter);
		long layerLength = layer.size();
		double* layerLinears = new double[layerLength];
		double* layerActivations = new double[layerLength];

		auto neuronIter = layer.begin();
		for (long i = 0; neuronIter != layer.end(); i++, neuronIter++)
		{
			tuple<double, double> neuronExecutionResults = (*neuronIter).ExecuteStore(networkActivations, ActivationFunction);
			layerLinears[i] = get<0>(neuronExecutionResults);
			layerActivations[i] = get<1>(neuronExecutionResults);
		}
		tuple<double*, double*> layerExecutionResults(layerLinears, layerActivations);
		return layerExecutionResults;
	}

	NeuralNetwork GetGradients(double* input)
	{

	}

	NeuralNetwork GetGradients(double** networkLinears, double** networkActivations)
	{

	}

	/// <summary>
	/// 
	/// </summary>
	/// <param name="layerI">Takes into account input layer as index 0</param>
	/// <param name="layerIterator"></param>
	/// <param name="neuronsGradient"></param>
	/// <param name="networkLinears"></param>
	/// <param name="networkActivations"></param>
	/// <returns></returns>
	list<Neuron> CalculateLayerGradients(long layerI, list<list<Neuron>>::iterator layerIterator, double** neuronsGradient, double** networkLinears, double** networkActivations)
	{
		list<Neuron> layer = (*layerIterator);
		list<Neuron> gradientsLayer = list<Neuron>();

		auto neuronIterator = layer.begin();
		long i = 0;
		while (neuronIterator != layer.end())
		{
			Neuron neuron = (*neuronIterator);
			tuple<double, list<double>, list<double>> gradients = neuron.GetGradients(networkActivations, networkLinears[layerI - 1][i], neuronsGradient[layerI][i], ActivationFunction);

			Neuron gradientNeuron = Neuron(get<0>(gradients), list<long>(), list<long>(), get<1>(gradients));

			NeuronConnectionsInfo neuronInfo = neuron.connections;
			auto connectionsXIterator = neuronInfo.Xs.begin();
			auto connectionsYIterator = neuronInfo.Ys.begin();
			auto activationGradientIterator = get<2>(gradients).begin();
			for (long j = 0; connectionsXIterator != neuronInfo.Xs.end(); j++, connectionsXIterator++, connectionsYIterator++, activationGradientIterator++)
			{
				neuronsGradient[*connectionsXIterator][*connectionsYIterator] -= (*activationGradientIterator);
			}

			gradientsLayer.push_back(gradientNeuron);
			neuronIterator++;
			i++;
		}
	}

public:
	ActivationFunctions::ActivationFunction GetActivationFunction()
	{
		return ActivationFunction;
	}

	/// <summary>
	/// Output includes input layer
	/// </summary>
	/// <returns></returns>
	list<long> GetNetworkShape()
	{
		long networkLength = GetNetworkLayerCount();
		list<long> shape = list<long>();
		shape.push_back(GetNetworkInputLength());

		auto layerIterator = Neurons.begin();
		for (long i = 0; layerIterator != Neurons.end(); i++, layerIterator++)
		{
			shape.push_back((*layerIterator).size());
		}
		return shape;
	}

	/// <summary>
	/// Output doesn't include input layer
	/// </summary>
	/// <returns></returns>
	long GetNetworkLayerCount()
	{
		return Neurons.size();
	}

	long GetNetworkInputLength()
	{
		auto layerIterator = Neurons.begin();
		auto neuronIterator = (*layerIterator).begin();
		return (*neuronIterator).connections.GetConnectionCount();
	}

	long GetNetworkOutputLength()
	{
		return OutputLength;
	}
};

