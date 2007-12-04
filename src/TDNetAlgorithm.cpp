/*
 *  TDNetAlgorithm.cpp
 *  TDNetsCPP
 *
 *  Created by Brian Tanner on 07/11/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */

#include "TDNetAlgorithm.h"

#include<iostream>
#include <fstream>

using namespace::std;

void TDNetAlgorithm::Save(ofstream &o) const{
	o<<" "<<netDepth<<" "<<historyStartParameter<<" "<<historyFeatureLength<<" "<<historyWeightsetLength<<" "<<useMCTraces<<" "<<nodeTypeToUse;
	theNet.Save(o);
}


TDNetAlgorithm::TDNetAlgorithm(ifstream &i, float stepSizeVal):
SolutionMethod(TDNetAlgSolType, stepSizeVal)
{
	//Read items from the file in the order we put them there
		int tmp;
		i>>netDepth>>historyStartParameter>>historyFeatureLength>>historyWeightsetLength>>useMCTraces;
		i>>tmp;
		nodeTypeToUse=(NodeFactory::NodeType)(tmp);
		
	//Now, lets get fancy and redo all the world of the other constructor.	
		theNet=TDNet(NodeFactory(nodeTypeToUse),i,allNodes);
		t=0;
		inputFact=inputPointerFactory(10000);		
		newTraces=new forgetfulTraceMemory(10000);
		oldTraces=new forgetfulTraceMemory(10000);
		tempTraces=NULL;
		stateHistory=forgetfulStateMemory(10000);
		y_t=NULL;
		y_tminus1=NULL;
		H=History(theNet.getActionCount(),theNet.getObsCount(),historyStartParameter,historyFeatureLength,historyWeightsetLength);
		pFact=predictionPointerFactory(theNet.getNodeCount(),10000);
	
		
		//Need to do an initial setup of the inputHistory
		//As part of the process, we can setup our input factory
		y_tminus1=pFact.getNextFreePointer();
		y_t=pFact.getNextFreePointer();
		
		//Put the info into this input history for future use
		tmpInput=inputFact.getNextFreePointer()->set(t,H,y_t);
		stateHistory.addToEnd(tmpInput);
		
//		cout<<"Printing TD Net: "<<endl;
//		cout<<theNet<<endl;
		
	
}

TDNetAlgorithm::TDNetAlgorithm(int actionCount, int observationCount, int nDepth, NodeFactory::NodeType typeToUse, char hStart, int hFLength,int hWLength, bool MCTrace, float alphaVal): 
SolutionMethod(TDNetAlgSolType, alphaVal),
netDepth(nDepth),
nodeTypeToUse(typeToUse),
theNet(TDNet(NodeFactory(nodeTypeToUse),netDepth,actionCount,observationCount)),
historyStartParameter(hStart),
historyFeatureLength(hFLength),
historyWeightsetLength(hWLength),
useMCTraces(MCTrace),
t(0),
inputFact(inputPointerFactory(10000)),
newTraces(new forgetfulTraceMemory(10000)),
oldTraces(new forgetfulTraceMemory(10000)),
tempTraces(NULL),
stateHistory(forgetfulStateMemory(10000)),
y_t(NULL),
y_tminus1(NULL),
H(History(actionCount,observationCount,historyStartParameter,historyFeatureLength,historyWeightsetLength)),
pFact(predictionPointerFactory(theNet.getNodeCount(),10000))
{
	theNet.addToGlobalVector(allNodes);
	theNet.createWeightsSets(H);
	
	//Add weights for the History into the TD Net Nodes
	for(vector<Node *>::const_iterator nIt=allNodes.begin();nIt!=allNodes.end();nIt++){
		(*nIt)->setupHistoryWeights(H.getFeatureSize());
	}
	
	theNet.hookNodesToEachOther();
	theNet.initAllWeights();
	
	//Need to do an initial setup of the inputHistory
	//As part of the process, we can setup our input factory
	y_tminus1=pFact.getNextFreePointer();
	y_t=pFact.getNextFreePointer();
	
	//Put the info into this input history for future use
	tmpInput=inputFact.getNextFreePointer()->set(t,H,y_t);
	stateHistory.addToEnd(tmpInput);
}


void TDNetAlgorithm::updateState(int a, int o){
	t++;

	H.update(a,o);
	
	//Swap y_t and y_tminus1
	y_tminus1=y_t;
	y_t=pFact.getNextFreePointer();
	
	
	//Calculate the new predictions
	for(int j=0;j<allNodes.size();j++){
		(*y_t)[j]=(allNodes[j]->calcPrediction(H,y_tminus1));
	}
	
	
	//Throw the tuple of the history and the previous predictions into the state history for future updates
	tmpInput=inputFact.getNextFreePointer()->set(t,H,y_tminus1);
	stateHistory.addToEnd(tmpInput);
	
	//Stop stateHistory from growing out of control
	if(stateHistory.size()>netDepth+2){
		stateHistory.popFront();
	}
	
	newTraces->clear();
	
	//Iterate over all of the traces in the heap
	while(oldTraces->size()>0){
		
		//Get the first trace element
		TraceNode *tmpTNode=oldTraces->popFront();
		
		
		//If the action is consistent with the question
		if(tmpTNode->checkCondition(a)){
			//Advance the parent up the tree
			tmpTNode->ageParent();
			//Select the appropriate set for previous input (stateHistory)
			int targetStep=t-tmpTNode->t;
			
			//Just a sanity check
			if(targetStep<1)
				cout<<"targetStep is less than 1, probably a problem"<<endl;
			
			
			//Learn
			if(tmpTNode->currentTarget==NULL||!useMCTraces)
				tmpTNode->learn(targetStep-1,stateHistory.getRelativeToEnd(targetStep),y_t,o);
			//				tmpTNode->cheatLearn(targetStep-1,stateHistory.getRelativeToEnd(targetStep),y_t,o,probAct0, probAct1);
			
			//If not at the top of the tree and lambda not 0, put back in for further updates
			if(tmpTNode->currentTarget!=NULL&&LAMBDA>=0.0f){
				if(useMCTraces){
					newTraces->setAtEnd(tmpTNode,tmpTNode->lastValue);
				}else{
					newTraces->setAtEnd(tmpTNode,(*y_t)[tmpTNode->currentTarget->globalIndex]);
				}
			}
		}//end of checkCondition
	}//End of passing through the traces
	
	//Put new stuff in with the traces
	for(int i=0;i<allNodes.size();i++){
		newTraces->setAtEnd(allNodes[i], allNodes[i], t, (*y_t)[i],(*y_t)[i], stepSize);
	}
	
	tempTraces=oldTraces;
	oldTraces=newTraces;
	newTraces=tempTraces;
	
	
}

float TDNetAlgorithm::getPForAO(int action, int observation) const{
	//For now, cheating, we know that node[a] has the prediction for a and they are predicting 1
	float p=(*y_t)[action];
	if(observation==0)
		p=1-p;
	return p;
}

TDNetAlgorithm::~TDNetAlgorithm(){
	delete(oldTraces);
	delete(newTraces);
	delete(y_t);
	delete(y_tminus1);
}
