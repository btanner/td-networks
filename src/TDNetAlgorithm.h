/*
 *  TDNetAlgorithm.h
 *  TDNetsCPP
 *
 *  Created by Brian Tanner on 07/11/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef TDNETALGORITHM_H
#define TDNETALGORITHM_H

#include "SolutionMethod.h"
#include "TDNet.h"
#include "NodeFactory.h"
#include "forgetfulTraceMemory.h"
#include "inputPointerFactory.h"
#include "forgetfulStateMemory.h"
#include "predictionPointerFactory.h"
#include <iostream>
using namespace::std;

class TDNetAlgorithm: public SolutionMethod{
public:
	
	
	TDNetAlgorithm(int actionCount, int observationCount, int nDepth, NodeFactory::NodeType typeToUse, char hStart, int hFLength, int hWLength, bool useMCTraces, float alpha);
	TDNetAlgorithm(ifstream &i, float stepSizeVal);
	~TDNetAlgorithm();
	
	virtual void Save(ofstream &o) const;

	
	
	virtual void updateState(int action, int observation);
	virtual float getPForAO(int action, int observation) const;

protected:
	/*Parameters we'll need to save.*/
	int netDepth;
	int historyFeatureLength;
	int historyWeightsetLength;
	char historyStartParameter;
	bool useMCTraces;
	NodeFactory::NodeType nodeTypeToUse;
	TDNet theNet;

	private:
	/*Parameters we'll need to recreate*/
	int t;
	float alpha;
	inputPointerFactory inputFact;
	forgetfulTraceMemory *newTraces;
	forgetfulTraceMemory *oldTraces;
	forgetfulTraceMemory *tempTraces;
	forgetfulStateMemory stateHistory;
	vector<float> *y_t;
	vector<float> *y_tminus1;

	//Create the global node structure, hook the nodes from the TD Net into it
	vector<Node *> allNodes;

	//Create the history structure
	History H;
	predictionPointerFactory pFact;
	Input *tmpInput;
};
#endif