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

#ifndef TDNETALGORITHM_H
#define TDNETALGORITHM_H

#include "SolutionMethod.h"
#include "TDNet.h"
#include "NodeFactory.h"
#include "forgetfulTraceMemory.h"
#include "inputPointerFactory.h"
#include "forgetfulStateMemory.h"
#include "predictionPointerFactory.h"
#include <iostream>
using namespace::std;

class TDNetAlgorithm: public SolutionMethod{
public:
	
	
	TDNetAlgorithm(int actionCount, int observationCount, int nDepth, NodeFactory::NodeType typeToUse, char hStart, int hFLength, int hWLength, bool useMCTraces, float alpha);
	TDNetAlgorithm(ifstream &i, float stepSizeVal);
	~TDNetAlgorithm();
	
	virtual void Save(ofstream &o) const;

	
	
	virtual void updateState(int action, int observation);
	virtual float getPForAO(int action, int observation) const;

protected:
	/*Parameters we'll need to save.*/
	int netDepth;
	int historyFeatureLength;
	int historyWeightsetLength;
	char historyStartParameter;
	bool useMCTraces;
	NodeFactory::NodeType nodeTypeToUse;
	TDNet theNet;

	private:
	/*Parameters we'll need to recreate*/
	int t;
	float alpha;
	inputPointerFactory inputFact;
	forgetfulTraceMemory *newTraces;
	forgetfulTraceMemory *oldTraces;
	forgetfulTraceMemory *tempTraces;
	forgetfulStateMemory stateHistory;
	vector<float> *y_t;
	vector<float> *y_tminus1;

	//Create the global node structure, hook the nodes from the TD Net into it
	vector<Node *> allNodes;

	//Create the history structure
	History H;
	predictionPointerFactory pFact;
	Input *tmpInput;
};
#endif