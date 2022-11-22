#pragma once
class NeuronStoredValues
{
public:
	//All neuron types

	float LinearFunction, OutputActivation;


	//LSTM and recurent

	float InitialHiddenState;
	float HiddenState;


	//LSTM

	float InitialCellState;
	float CellState;

	//Hidden state

	float HiddenStateSigmoid;
	float HiddenStateTanh;


	//Gates:
	
	//	Forget

	float ForgetWeightMultiplication;
	float ForgetGateMultiplication;


	//	Store

	float StoreSigmoidWeightMultiplication;
	float StoreTanhWeightMultiplication;
	float StoreGateMultiplication;
	float StoreGateAddition;


	//	Output
	float CellStateTanh;
	float OutputWeightMultiplication;
};

