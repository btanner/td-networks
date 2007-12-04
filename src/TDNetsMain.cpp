/* TD-Networks C++ Implementation
* Copyright (C) 2007, Brian Tanner brian@tannerpages.com (http://brian.tannerpages.com/)
* This code can be found at http://td-networks.googlecode.com
* 
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License
* as published by the Free Software Foundation; either version 2
* of the License, or (at your option) any later version.
* 
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
* 
* You should have received a copy of the GNU General Public License
* along with this program; if not, write to the Free Software
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA. */

#include <iostream>
#include <vector>
#include <stdio.h>
#include <sstream> 
#include <string.h>

#include "TDNet.h"
#include "ringWorldEnv.h"
#include "cycleWorldEnv.h"
#include "TwoDimensionalPOMDP.h"
#include "TwoDimensionalEnv.h"
#include "TwoDimensionalMap.h"
#include "History.h"
#include "node.h"
#include "TraceNode.h"
#include "Input.h"
#include "predictionPointerFactory.h"
#include "inputPointerFactory.h"
#include "forgetfulStateMemory.h"
#include "forgetfulTraceMemory.h"
#include "staticInits.h"
#include "NodeFactory.h"

#include "TDNetAlgorithm.h"


using std::cout;
using std::endl;
using std::vector;
using std::string;

//This runs an algorithm for a fixed number of steps
float runExperiment(int totalSteps, int stepsPerBin, SolutionMethod *Alg,Environment *Env,bool printOutput=false);
//This runs an experiment to an error tolerance
float runExperimentTol(float tol, int stepsPerBin, SolutionMethod *Alg,Environment *Env,bool printOutput,int &numSteps, int &lastNumWrong);

int main (int argc, char * const argv[]) {
	int totalSteps=1000000;
	int stepsPerBin=25000;
	
	//Pointers for the Environment we'll be in and the solution method we'll be using
	Environment *Env=NULL;
	SolutionMethod *Alg=NULL;

	//This creates a new 8-state ring world environment
	Env = new ringWorldEnv(5);
	
	//When using TD Nets + history if length k (and k is odd), it matters if you are starting the history on actions or observations
	//This can be 'a' or 'o'
	char historyStartParam='a';
	
	//Learning rate
	float alpha=.5f;
	
	//Use Monte Carlo networks with eligibility traces.  If you set this to false you'll use Vanilla TD-Networks
	bool useMCTraces=true;

//Choose what types of nodes you'll be using.  I can't remember if you can mix and match.
//The literature out there used SigmoidOneStep Nodes.  We later found (but never thoroughly investigated that cross entropy nodes were better)
//They just differ in their activation function and of course the inverse activation
//	NodeFactory::NodeType typeofNode=NodeFactory::CrossEntropyOneStep; 	//activate(float P){return(1.0/(1.0+exp(-P)));}
	NodeFactory::NodeType typeofNode=NodeFactory::SigmoidOneStep;		//activate(float P){return(P*(1.0f-P));}
//	NodeFactory::NodeType typeofNode=NodeFactory::OneStep;				//activate(float P) :: returns P truncated to be in [0,1]
	
	//This will create a TDNet Learning Agent/Algorithm
	Alg=new TDNetAlgorithm(Env->getActionCount(),
						   Env->getObservationCount(),
						   3,
						   typeofNode,
						   historyStartParam,
						   0,
						   2,
						   useMCTraces,
						   alpha);	

				int lastNumWrong,numSteps;
				float results=runExperimentTol(0.0001f,stepsPerBin,Alg,Env,true,numSteps,lastNumWrong);
				//	cout<<"Error with weight Feature length: "<<historyFeatureLength<<" weight Set length: "<<historyWeightsetLength<<" and depth: "<<Depth<< " is: "<<results<<endl;
				//	cout<<" Last wrong "<<lastNumWrong<<" number of steps "<<numSteps<<endl;
				exit(1);
				
//
//
//	The code below this is used to run multiple experiments with different parameters and then save them to some files
//	I can't remember exactly how it works.
//
//
				
				ofstream allResultsFile;
				ofstream bestResultsFile;
				ofstream batchesFile;
				allResultsFile.open("../../results/allResults");
				bestResultsFile.open("../../results/bestResults");
				batchesFile.open("../../results/batches");
				
				bestResultsFile <<"["<<endl;
				batchesFile <<"["<<endl;
				
				
				int numRuns=1;
				
				int depths[] = {3};
				int features[] = {0};
				int weights[]={4};
				
				for(int i=0; i<numRuns; i++)
				{
					int Depth = depths[i];
					int historyFeatureLength = features[i];
					int historyWeightsetLength = weights[i];
					int hFL = historyFeatureLength;
					int hWL = historyWeightsetLength;
					int d = Depth;
					
					for(int j=0; j<1; j++)
					{
						Alg=new TDNetAlgorithm(Env->getActionCount(),
											   Env->getObservationCount(),
											   Depth,
											   typeofNode,
											   historyStartParam,
											   historyFeatureLength,
											   historyWeightsetLength,
											   useMCTraces,
											   alpha);	
						
						//				float results=runExperiment(totalSteps,stepsPerBin,Alg,Env,true);
						
						int numSteps, lastNumWrong;
						float results=runExperimentTol(0.001f,stepsPerBin,Alg,Env,false,numSteps,lastNumWrong);
						allResultsFile<<"it "<<j<<" history Feature length: "<<hFL<<" weight Set length: "<<hWL<<" and depth: "<<d<< " is: "<<results<<" with "<<lastNumWrong<<" after "<<numSteps<<" iterations."<<endl;
						cout<<"Error hFL: "<<hFL<<" hWL: "<<hWL<<" d: "<<d<< " is: "<<results<<" wrong: "<<lastNumWrong<<" after it: "<<numSteps<<endl;
						
						std::string dir("../../results/");
						std::ostringstream oss;
						oss << dir << "it"<<j<<"hFL"<<hFL<<"hWL"<<hWL<<"d"<<d;
						std::string fileName(oss.str());
						//cout << fileName;
						
						Alg->saveToFile(fileName);
						
						delete(Alg);
						cout << "Results: "<<results<<endl;
						bestResultsFile << " "<<results;
						batchesFile <<" "<<numSteps;
					}
					bestResultsFile <<";"<<endl;
					batchesFile <<";"<<endl;
				}//end iterating over depth
				
				allResultsFile.close();
				bestResultsFile << "]"<<endl;
				bestResultsFile.close();
				
				batchesFile << "]"<<endl;
				batchesFile.close();
				return 0;
}

float runExperiment(int totalSteps, int stepsPerBin, SolutionMethod *Alg,Environment *Env,bool printOutput){
//You can write code so that you can load previous experiments from a file and see how they do
	//SolutionMethod* Alg=SolutionMethod::loadFromFile("data.dat");


//The current action and observation
	int a;
	int o;
	
//The error that observed between prediction and observation
	float observedError=0.0f;
	
	float finalBinError=0.0f;
	
	int t=0;
	for(int i=0;i<totalSteps;i++){
		t++;
		
//This code prints out the error at regular intervals
		if(i%stepsPerBin==0){
			if(printOutput){
				cout<<"OE: "<<observedError/float(stepsPerBin)<<"\tbin: "<<i/stepsPerBin<<endl;
			}
			finalBinError=observedError/float(stepsPerBin);
			observedError=0.0f;
//Apparently we half the step size at this interval.  I didn't expect that.  You can try without this
			Alg->setStepSize(Alg->getStepSize()*.5);
		}
		
//Get details about this step from the environment (we actually advance it to the next state later)
//We're trying to learn to predict the environment, not control it, so we ask the environment for the action
//This method has no side effect, so you can forgo this and use whatever action selection mechanism you want
		a=Env->nextAction();		
		o=Env->getObservation(a);
		
		
//Save results occasionally
		if(i%(stepsPerBin*25)==0&&i>0){
			if(DET)
				cout<<"Saving to file..."<<endl;
			Alg->saveToFile("b2bLearning.dat");
		}
		
//Looks like our performance criterion is absolute error
		observedError+=fabs(1.0f-Alg->getPForAO(a,o));
		
//Advance the Alg and Env to the next state
		Alg->updateState(a,o);
		Env->nextState(a,o);
		
	}//End of the time step
	Alg->saveToFile("b2bLearning.dat");
	return finalBinError;
}

float runExperimentTol(float tol, int stepsPerBin, SolutionMethod *Alg,Environment *Env,bool printOutput, int &numSteps, int &lastNumWrong){
	//SolutionMethod* Alg=SolutionMethod::loadFromFile("data.dat");
	int a;
	int o;
	
	float observedError=0.0f;
	
	float prevBinError = 4*tol;
	float curBinError = 2*tol;
	float runningBinError = 0.0f;
	
	float prevFinalBinError = 0.0f;
	float finalBinError=0.0f;
	
	int numWrong = 0;
	
	int binsPerTol = 20;
	
	int t=0;
	int i=0;
	while((prevBinError-curBinError)>tol){
		t++;
		i++;
		
		if(i%stepsPerBin==0){
			if(printOutput){
				cout<<"OE: "<<observedError/float(stepsPerBin)<<"\t numWrong: "<<numWrong<<"\tbin: "<<i/stepsPerBin<<endl;
			}
			lastNumWrong = numWrong;
			numWrong = 0;
			
			prevFinalBinError = finalBinError;
			finalBinError=observedError/float(stepsPerBin);
			if( prevFinalBinError == 0.0f)
				prevFinalBinError = finalBinError+2*tol;
			
			runningBinError = runningBinError + prevFinalBinError - finalBinError;
			//cout <<" prevFinalBinError "<<prevFinalBinError<< " finalBinError: "<<finalBinError<<"runningBinError: "<<runningBinError<<endl;
			observedError=0.0f;
			
			if(i%(stepsPerBin*binsPerTol)==0){
				prevBinError = curBinError;
				curBinError = runningBinError;
				
				runningBinError = 0.0f;
				if(i/(stepsPerBin*binsPerTol)==1)
					prevBinError = curBinError+2*tol;
				
				//cout <<"curBinError: "<<curBinError<<" prevBinError: "<<prevBinError<<" Diff: "<<prevBinError-curBinError<<endl;
			}
		}
		
		a=Env->nextAction();		
		o=Env->getObservation(a);
		
		float err = 1.0f-Alg->getPForAO(a,o);
		observedError+=fabs(err);
		if( err > .5 )
			numWrong++;
		
		Alg->updateState(a,o);
		Env->nextState(a,o);
		
		if(i%(stepsPerBin*25)==0&&i>0){
			if(DET)
				cout<<"Saving to file..."<<endl;
			Alg->saveToFile("data.dat");
			Alg->setStepSize(Alg->getStepSize()*.5);
		}
		
	}//End of the time step
	numSteps = i;
	return finalBinError;
}