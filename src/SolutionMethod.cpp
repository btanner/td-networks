/*
 *  SolutionMethod.cpp
 *  TDNetsCPP
 *
 *  Created by Brian Tanner on 07/11/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */

#include "SolutionMethod.h"
#include "TDNetAlgorithm.h"
#include <fstream>
#include <string>

using std::string;

SolutionMethod::SolutionMethod(SolutionType theType, float stepSizeVal):
theSolutionType(theType), stepSize(stepSizeVal){}

SolutionMethod::~SolutionMethod(){}


void SolutionMethod::saveToFile(string fileName) const{
	ofstream outFile;
	outFile.open(fileName.c_str(),std::ios::binary);
	outFile<<theSolutionType<<" "<<stepSize;
	Save(outFile);
	
	outFile.close();
}

float SolutionMethod::getStepSize() const{
	return stepSize;
}

void SolutionMethod::setStepSize(float stepSizeVal){
	stepSize=stepSizeVal;
}


SolutionMethod *SolutionMethod::loadFromFile(string fileName) {
	SolutionMethod *rSolution;
	ifstream inFile;
	float stepSize;
	inFile.open(fileName.c_str(),std::ios::binary);
	int theType;
	inFile>>theType>>stepSize;
	SolutionType theSolutionType=(SolutionType)theType;
	if(theSolutionType==TDNetAlgSolType){
		cout<<"The file says its a TDNetAlgSol!"<<endl;
		rSolution=new TDNetAlgorithm(inFile, stepSize);
	}else{
		cout<<"Tried to load an algorithm from a stored representation that is of unknown type."<<endl;
	}
	
	inFile.close();
	return rSolution;
}

