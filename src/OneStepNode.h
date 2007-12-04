/*
 *  Node.h
 *  TDNetsCPP
 *
 *  Created by Brian Tanner on 23/08/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */
 
#ifndef ONESTEPNODE_H
#define ONESTEPNODE_H

#include <ostream>
#include <fstream>
#include <string>
#include <vector>
#include "Printable.h"
#include "Weights.h"
#include "History.h"
#include "Input.h"
#include <cmath>
#include "Node.h"

using namespace::std;

#define PARENT_OBS NULL;
class OneStepNode : public Node{
public:

	OneStepNode(int obsToPredict, int actCond, Node *parent, int mcTargetTimeStep);
	OneStepNode(ifstream &i);
	
	//Functions were are contractually obligated to implement
	virtual void Save(ofstream &o)const;

	virtual string toString() const;
	virtual string toStringExtended() const;
	
	virtual void initAllWeights();
	virtual void createWeightsSets(History &H);
	
	virtual void setupHistoryWeights(int historySize);
	virtual float calcPrediction(History &H,vector<float> *y_tminus1) const;
	
	virtual float activate(float P) const;
	virtual float inverseActivate(float P) const;

	//Inline functions
	virtual Node *getParent() const;
	virtual int getObsToPredict() const;
	virtual int getActCond() const;
	virtual void setParent(Node *p);
	virtual void fixLinksAfterLoadFromFile(vector<Node *> &allNodes);

	virtual void updateNode(float alphaDelta, const Input *theInput);
	virtual void addPredictionWeight(Node *theNode);

private:
	vector<Weights> theWeights;
	int actCond;
	int obsToPredict;
	Node *parent;
	int mcTargetTimeStep;
	
};
#endif