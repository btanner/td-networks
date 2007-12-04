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

