/*
 *  NodeFactory.h
 *  TDNetsCPP
 *
 *  Created by Brian Tanner on 23/09/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef NODEFACTORY_H
#define NODEFACTORY_H

#include "Node.h"
#include <fstream>

using namespace::std;

class NodeFactory{
public:
enum NodeType
{
	OneStep,
	SigmoidOneStep,
	CrossEntropyOneStep
};

NodeFactory(NodeType typeInit=OneStep);
Node *createNode(int actCond, int obsToPredict, Node *parent, int mcTargetTimeStep);
Node *createNode(ifstream &f);

private:
NodeType  theType;
};

#endif