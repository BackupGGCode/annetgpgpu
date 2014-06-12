/*
 * main.cpp
 *
 *  Created on: 12.04.2010
 *      Author: dgrat
 */

#include <ANNet>
#include <ANContainers>
#include <ANMath>

#include "Samples.h"

#include <ctime>
#include <iostream>


int main(int argc, char *argv[]) {
	ANN::BPNet cpu_one;

	ANN::BPLayer layer1(3, ANN::ANLayerInput);
	ANN::BPLayer layer2(32, ANN::ANLayerHidden);
	ANN::BPLayer layer3(6, ANN::ANLayerOutput);

	layer1.ConnectLayer(&layer2);
	layer2.ConnectLayer(&layer3);

	cpu_one.AddLayer(&layer1);
	cpu_one.AddLayer(&layer2);
	cpu_one.AddLayer(&layer3);

	ANN::TrainingSet input;
	input.AddInput(fInp1, 3);
	input.AddOutput(fOut1, 6);
	input.AddInput(fInp2, 3);
	input.AddOutput(fOut2, 6);
	input.AddInput(fInp3, 3);
	input.AddOutput(fOut3, 6);
	input.AddInput(fInp4, 3);
	input.AddOutput(fOut4, 6);
	input.AddInput(fInp5, 3);
	input.AddOutput(fOut5, 6);
	input.AddInput(fInp6, 3);
	input.AddOutput(fOut6, 6);
	input.AddInput(fInp7, 3);
	input.AddOutput(fOut7, 6);
	input.AddInput(fInp8, 3);
	input.AddOutput(fOut8, 6);
	input.AddInput(fInp9, 3);
	input.AddOutput(fOut9, 6);
	input.AddInput(fInp10, 3);
	input.AddOutput(fOut10, 6);
	
	std::vector<float> errors;
	cpu_one.SetLearningRate(0.075);
	cpu_one.SetMomentum(0);
	cpu_one.SetWeightDecay(0);
	cpu_one.SetTrainingSet(input);

	bool b = false;
	float f;
	errors = cpu_one.TrainFromData(10000, 0.001, b, f);
	std::cout<< cpu_one <<std::endl;

	cpu_one.ExpToFS("foo.bar");
	cpu_one.ImpFromFS("foo.bar");

	cpu_one.SetTrainingSet(input);
	std::cout<< cpu_one <<std::endl;

	return 0;
}
