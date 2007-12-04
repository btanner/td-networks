/*
 *  CrossEntropyOneStepNode.cpp
 *  TDNetsCPP
 *
 *  Created by Brian Tanner on 23/09/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */

#include "CrossEntropyOneStepNode.h"
#include <cmath>
float CrossEntropyOneStepNode::activate(float P) const{
	return(1.0/(1.0+exp(-P)));	
}

CrossEntropyOneStepNode::CrossEntropyOneStepNode(int actCond, int obsToPredict, Node *parent, int mcTargetTimeStep):
OneStepNode(actCond,obsToPredict,parent,mcTargetTimeStep){}
