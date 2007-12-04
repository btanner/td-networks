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
 
#ifndef NODE_H
#define NODE_H
#include "Serializable.h"
#include <fstream>
#include <ostream>
#include <string>
#include <vector>
#include "Printable.h"
#include "Weights.h"
#include "History.h"
#include "Input.h"
#include <cmath>
/*
 Perhaps each node should have a vector of weights, inside the node. Now things are really distributed.
 Each weight could have a lookup field associated with, for which feature it corresponds to.  This way, 
 the feature vector doesn't have to be the same as the weight vector for a node.
 
 We could also then randomly add weights (and remove weights) from a node.  This could be great.
 
 However, the problem may remain, how do we do things like history, where there is only one that is active.
 
 We'll still have to split them into categories.  Perhaps we'll have independent weight vectors for each feature
 type.
 */
using namespace::std;

#define PARENT_OBS NULL;
class Node : public Printable, public Serializable{
public:
	int globalIndex;
	//used ONLY for loading from files...
	int __parentID;
	
	static const int COND_ANY=-1;
	
	
	//Interface calls
	virtual string toString() const=0;
	virtual void Save(ofstream &o) const=0;

	virtual void fixLinksAfterLoadFromFile(vector<Node *> &allNodes)=0;
	
	virtual string toStringExtended() const=0;
	
	virtual void initAllWeights()=0;
	virtual void createWeightsSets(History &H)=0;
	
	virtual void setupHistoryWeights(int historySize)=0;
	virtual float calcPrediction(History &H,vector<float> *y_tminus1) const=0;
	
	virtual float activate(float P) const = 0;
	virtual float inverseActivate(const float P) const = 0;
	
	virtual Node *getParent() const=0;
	virtual void setParent(Node *p) =0;
	virtual int getObsToPredict() const=0;
	virtual int getActCond() const=0;

	virtual void updateNode(float alphaDelta, const Input *theInput)=0;
	virtual void addPredictionWeight(Node *theNode)=0;
};


#endif