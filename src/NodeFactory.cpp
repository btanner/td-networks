/* TD-Networks C++ Implementation
* Copyright (C) 2007, Brian Tanner brian@tannerpages.com (http://brian.tannerpages.com/)
* This code can be found at http://td-networks.googlecode.com
* 
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License
* as published by the Free Software Foundation; either version 2
* of the License, or (at your option) any later version.
* 
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
* 
* You should have received a copy of the GNU General Public License
* along with this program; if not, write to the Free Software
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA. */

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
