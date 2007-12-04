/*
 *  SigmoidOneStepNode.cpp
 *  TDNetsCPP
 *
 *  Created by Brian Tanner on 22/09/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */

#include "SigmoidOneStepNode.h"
#include "math.h"
#include <iostream>

using std::cout;
using std::endl;

float SigmoidOneStepNode::inverseActivate(const float P) const{
	return (P*(1.0f-P));
}

SigmoidOneStepNode::SigmoidOneStepNode(int actCond, int obsToPredict, Node *parent, int mcTargetTimeStep):
CrossEntropyOneStepNode(actCond,obsToPredict,parent,mcTargetTimeStep){}
