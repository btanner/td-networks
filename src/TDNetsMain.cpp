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


/*This code needs to be updated to not have all the POMDP conversion stuff in it */

using std::cout;
using std::endl;
using std::vector;
using std::string;

float runExperiment(int totalSteps, int stepsPerBin, SolutionMethod *Alg,Environment *Env,bool printOutput=false);
float runExperimentTol(float tol, int stepsPerBin, SolutionMethod *Alg,Environment *Env,bool printOutput,int &numSteps, int &lastNumWrong);

int main (int argc, char * const argv[]) {
	int totalSteps=1000000;
	int stepsPerBin=25000;
	cout<<"Starting program"<<endl;
	
	Environment *Env=NULL;
	SolutionMethod *Alg=NULL;

	Env = new ringWorldEnv(8);
	
	char historyStartParam='a';
	float alpha=.5f;
	
	bool useMCTraces=true;
//	NodeFactory::NodeType typeofNode=NodeFactory::CrossEntropyOneStep;
	NodeFactory::NodeType typeofNode=NodeFactory::SigmoidOneStep;
//	NodeFactory::NodeType typeofNode=NodeFactory::OneStep;
	
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
				float results=runExperimentTol(0.01f,stepsPerBin,Alg,Env,true,numSteps,lastNumWrong);
				//	cout<<"Error with weight Feature length: "<<historyFeatureLength<<" weight Set length: "<<historyWeightsetLength<<" and depth: "<<Depth<< " is: "<<results<<endl;
				//	cout<<" Last wrong "<<lastNumWrong<<" number of steps "<<numSteps<<endl;
				exit(1);
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
	//SolutionMethod* Alg=SolutionMethod::loadFromFile("data.dat");
	int a;
	int o;
	
	float observedError=0.0f;
	
	float finalBinError=0.0f;
	
	int t=0;
	for(int i=0;i<totalSteps;i++){
		t++;
		
		if(i%stepsPerBin==0){
			if(printOutput){
				cout<<"OE: "<<observedError/float(stepsPerBin)<<"\tbin: "<<i/stepsPerBin<<endl;
			}
			finalBinError=observedError/float(stepsPerBin);
			observedError=0.0f;
			Alg->setStepSize(Alg->getStepSize()*.5);
		}
		
		a=Env->nextAction();		
		o=Env->getObservation(a);
		
		
		if(i%(stepsPerBin*25)==0&&i>0){
			if(DET)
				cout<<"Saving to file..."<<endl;
			Alg->saveToFile("b2bLearning.dat");
		}
		
		observedError+=fabs(1.0f-Alg->getPForAO(a,o));
		
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