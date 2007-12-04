/*
 *  HistoryAlgorithm.cpp
 *  HistorysCPP
 *
 *  Created by Brian Tanner on 07/11/05.
 *  Modified by Anna Koop on 16/11/05
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */

#include "HistoryAlgorithm.h"

#include<iostream>
#include <fstream>

using namespace::std;

void HistoryAlgorithm::Save(ofstream &o) const{
	o<<" "<<historyLength<<" ";
	//need to save weights
}


HistoryAlgorithm::HistoryAlgorithm(ifstream &i):
SolutionMethod(HistoryAlgSolType, 1)
{
	//need to save to file
}

HistoryAlgorithm::HistoryAlgorithm(int aCount, int oCount, int hLength): 
	SolutionMethod(HistoryAlgSolType, 1),
	historyLength(hLength),
	t(0),
	actionCount(aCount),
	observationCount(oCount),
//	stateHistory(forgetfulStateMemory(10000)),
	H(History(actionCount,observationCount,'a',historyLength,historyLength))
{
	numH = new float[H.getFeatureSize()];
	numOH = new float[H.getFeatureSize()*observationCount];
	//create the array for storing predictions
	//initialize all weights to 0
	for(int i=0; i<H.getFeatureSize(); i++) {
		numH[i] = 0.0f;
	}
	
	for(int i=0; i<H.getFeatureSize()*observationCount; i++) {
		numOH[i] = 0.0f;
	}
		
	//Put the info into this input history for future use
//	tmpInput=inputFact.getNextFreePointer()->set(t,H,y_t);
//	stateHistory.addToEnd(tmpInput);
}


void HistoryAlgorithm::updateState(int a, int o){
	t++;

	H.updateAction(a);
	int historyIndex = H.getFeatureIndex();
	int ohIndex = observationCount*historyIndex+o;

	numH[historyIndex]+=1.0f;
	numOH[ohIndex]+=1.0f;
	H.updateObservation(o);
}

//historyIndex needs to be last action backwards
float HistoryAlgorithm::getPForAO(int action, int observation) const{
	int historyIndex, ohi, a;
	History temp = H;

	a = temp.updateAction(action);
	historyIndex = temp.getFeatureIndex();

	ohi = observationCount*historyIndex+observation;
	//cheating knowing that observations start at 0
	if(numH[historyIndex] == 0.0f )
		return 0.0f;
	else {
		float num = numOH[ohi];
		float denom = numH[historyIndex];
		float prob = numOH[ohi]/numH[historyIndex];
		return prob;
	}
}

HistoryAlgorithm::~HistoryAlgorithm(){
	delete(numH);
	delete(numOH);
}

