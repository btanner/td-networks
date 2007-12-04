/*
 *  SigmoidOneStepNode.h
 *  TDNetsCPP
 *
 *  Created by Brian Tanner on 22/09/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef SIGMOIDONESTEPNODE_H
#define SIGMOIDONESTEPNODE_H
#include "CrossEntropyOneStepNode.h"

#include <fstream>
using namespace::std;

class  SigmoidOneStepNode : public CrossEntropyOneStepNode{
public:
	virtual float inverseActivate(const float P) const;
	SigmoidOneStepNode(int obsToPredict, int actCond, Node *parent, int mcTargetTimeStep);
	SigmoidOneStepNode(ifstream &i);
};

inline SigmoidOneStepNode::SigmoidOneStepNode(ifstream &i):CrossEntropyOneStepNode(i){}
#endif
