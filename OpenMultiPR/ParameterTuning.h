#pragma once

// OpenGA library
#include "genetic.hpp"
#include "Header.h"
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "SequenceSearch.h"

struct Params
{
	float lambda[10] = { 2.24987526,	0.39806605	,2.02482809,	0.49958895	,1.16373923,	0.63717877	,0.66855686,	1.53769591	,0.82047087 , 1};
	float vmax = 2.86, vmin = 0.35, numsequence = 45;
	float scoreTh = 0;
};

// Genetic Algorithm
struct MyGenes
{
	std::vector<double> x; //paramters to be optimized: lambda

	std::string to_string() const
	{
		std::ostringstream out;
		out << "{";
		for (unsigned long i = 0; i < x.size(); i++)
			out << (i ? "," : "") << std::setprecision(10) << x[i];
		out << "}";
		return out.str();
	}
};
struct MyMiddleCost
{
	// This is where the results of simulation
	// is stored but not yet finalized.
	double cost;
};

class Parameter2F1
{
	std::vector<std::vector<int>> gt;
	std::vector<int> pGlobal[10];
	Params parameters;
	std::vector<int> matchingResults; // -1 means empty, 1-based results
	cv::Size matSize;
	SequenceSearch pSS[10];
public:
	//Parameter2F1(){};
	Parameter2F1(std::vector<std::vector<int>> gt, std::vector<int> GGGlobalBest, std::vector<int> BoWGlobalBest_RGB, std::vector<int> BoWGlobalBest_D, std::vector<int> BoWGlobalBest_IR,
		std::vector<int> GISTGlobalBest_RGB, std::vector<int> GISTGlobalBest_D, std::vector<int> GISTGlobalBest_IR,
		std::vector<int> LDBGlobalBest_RGB, std::vector<int> LDBGlobalBest_D, std::vector<int> LDBGlobalBest_IR, cv::Size matSize) : gt(gt), matSize(matSize)
	{
		pGlobal[0] = BoWGlobalBest_RGB;
		pGlobal[1] = BoWGlobalBest_D;
		pGlobal[2] = BoWGlobalBest_IR;
		pGlobal[3] = GISTGlobalBest_RGB;
		pGlobal[4] = GISTGlobalBest_D;
		pGlobal[5] = GISTGlobalBest_IR;
		pGlobal[6] = LDBGlobalBest_RGB;
		pGlobal[7] = LDBGlobalBest_D;
		pGlobal[8] = LDBGlobalBest_IR;
		pGlobal[9] = GGGlobalBest;
	};
	~Parameter2F1() {	};
	//update different parameters
	void updateParams(std::vector<double> x){
		for (size_t i = 0; i < 10; i++)
		{
			parameters.lambda[i] = x[i];
		}
	}
	void updateParams(float vmax, float vmin, float numseq)
	{
		parameters.vmax = vmax;
		parameters.vmin = vmin;
		parameters.numsequence = numseq;
	}
	void updateParams(float scoreTh)
	{
		parameters.scoreTh = scoreTh;
	}


	void prepare4MultimodalCoefficients();	
	// get matching results
	void placeRecognition();
	float placeRecognition4MultimodalCoefficients(const MyGenes& p);

	void printMatchingResults() {
		for (size_t i = 0; i < matchingResults.size(); i++)
		{
			std::cout << i << "..." << matchingResults[i] << std::endl;
		}
	}
	//calculate F1 score according to groundtruth(gt) and PR results (matchingResults).
	float calculateF1score();
	float calculateErr();

	bool eval_genes(const MyGenes& p, MyMiddleCost &c);
};

typedef EA::Genetic<MyGenes, MyMiddleCost> GA_Type;
typedef EA::GenerationType<MyGenes, MyMiddleCost> Generation_Type;

// functions of GA 
void init_genes(MyGenes& p, const std::function<double(void)> &rand);

MyGenes mutate(const MyGenes& X_base, const std::function<double(void)> &rand, double shrink_scale);
MyGenes crossover(const MyGenes& X1, const MyGenes& X2, const std::function<double(void)> &rand);
double calculate_SO_total_fitness(const GA_Type::thisChromosomeType &X);
void SO_report_generation(int generation_number, const EA::GenerationType<MyGenes, MyMiddleCost> &last_generation, const MyGenes& best_genes);

// execute optimization for parameters lambda
bool optimizeMultimodalCoefficients(Parameter2F1* pt, std::vector<double>& x);

