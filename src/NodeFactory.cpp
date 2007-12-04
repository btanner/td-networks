/*
 *  NodeFactory.cpp
 *  TDNetsCPP
 *
 *  Created by Brian Tanner on 23/09/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */

#include "NodeFactory.h"

#include "Node.h"
#include "OneStepNode.h"
#include "CrossEntropyOneStepNode.h"
#include "SigmoidOneStepNode.h"
#include <iostream>
#include <fstream>

using namespace std;

NodeFactory::NodeFactory(NodeType typeInit): theType(typeInit){}

Node *NodeFactory::createNode(ifstream &i){
	if(theType==OneStep)
		return new OneStepNode(i);
	if(theType==SigmoidOneStep)
		return new SigmoidOneStepNode(i);
	if(theType==CrossEntropyOneStep)
		return new CrossEntropyOneStepNode(i);
	cout<<"Was asked to create a node type that I don't understand.  Quitting."<<endl;
	exit(1);
}

Node *NodeFactory::createNode(int actCond, int obsToPredict, Node *parent, int mcTargetTimeStep){
	if(theType==OneStep)
		return new OneStepNode(actCond,obsToPredict, parent, mcTargetTimeStep);
	if(theType==SigmoidOneStep)
		return new SigmoidOneStepNode(actCond,obsToPredict, parent, mcTargetTimeStep);
	if(theType==CrossEntropyOneStep)
		return new CrossEntropyOneStepNode(actCond,obsToPredict, parent, mcTargetTimeStep);
	
	cout<<"Was asked to create a node type that I don't understand.  Quitting."<<endl;
	exit(1);

	

}
