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