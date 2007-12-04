/*
 *  CrossEntropyOneStepNode.h
 *  TDNetsCPP
 *
 *  Created by Brian Tanner on 23/09/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef CROSSENTROPYONESTEPNODE_H
#define CROSSENTROPYONESTEPNODE_H
#include "OneStepNode.h"
#include <fstream>

using namespace std;
class  CrossEntropyOneStepNode : public OneStepNode{
public:
	virtual float activate(float P) const;
	CrossEntropyOneStepNode(int obsToPredict, int actCond, Node *parent, int mcTargetTimeStep);	
	CrossEntropyOneStepNode(ifstream &i);
};

inline CrossEntropyOneStepNode::CrossEntropyOneStepNode(ifstream &i):OneStepNode(i){}

#endif