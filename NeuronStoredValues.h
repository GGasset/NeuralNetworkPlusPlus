#pragma once
class NeuronStoredValues
{
public:
	//All neuron types:

	float LinearFunction, OutputActivation;


	//LSTM and recurent:

	float InitialHiddenState;
	float HiddenLinear;
	float HiddenState;


	//LSTM:

	float InitialCellState;
	float CellState;

	//	Hidden state calculations:

	float HiddenLinearSigmoid;
	float HiddenLinearTanh;


	//	Gates:
	
	//		Forget

	float ForgetWeightMultiplication;
	float ForgetGateMultiplication;


	//		Store

	float StoreSigmoidWeightMultiplication;
	float StoreTanhWeightMultiplication;
	float StoreGateMultiplication;
	float StoreGateAddition;


	//		Output
	float CellStateTanh;
	float OutputWeightMultiplication;
};

