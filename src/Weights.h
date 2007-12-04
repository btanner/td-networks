/*
 *  Weights.h
 *  TDNetsCPP
 *
 *  Created by Brian Tanner on 01/09/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */


#ifndef WEIGHTS_H
#define WEIGHTS_H

#include "Printable.h"
#include "Serializable.h"
#include "Input.h"
#include "History.h"
#include <vector>
#include <string>


using std::vector;
using std::string;
class Node;

/* Its unclear how this will work */
class Weights : public  Printable {
public:
	Weights();

	virtual string toString() const;

	//Stuff we have to implement
	virtual void Save(ofstream &o)const;
	void fixLinksAfterLoadFromFile(vector<Node *> &allNodes);
	virtual void Load(ifstream &i);

	void updateWeights(float alphaDelta, const Input *theInput);
	float weightedSum(History &H,vector<float> *y_tminus1) const;
	void initWeights();
	void addPredictionWeight(Node *theNode);
	void setupHistoryWeights(int historySize);


	vector<float> historyWeights;
	vector<float> predictionWeights;
	vector<Node *> pWLookup;
//This is totally redundant and is only useful when loading things from file.  It might not even be set otherwise.
	vector<int> __pwIDs;
	
	float biasWeight;
};

#endif