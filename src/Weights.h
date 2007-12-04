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


#ifndef WEIGHTS_H
#define WEIGHTS_H

#include "Printable.h"
#include "Serializable.h"
#include "Input.h"
#include "History.h"
#include <vector>
#include <string>


using std::vector;
using std::string;
class Node;

/* Its unclear how this will work */
class Weights : public  Printable {
public:
	Weights();

	virtual string toString() const;

	//Stuff we have to implement
	virtual void Save(ofstream &o)const;
	void fixLinksAfterLoadFromFile(vector<Node *> &allNodes);
	virtual void Load(ifstream &i);

	void updateWeights(float alphaDelta, const Input *theInput);
	float weightedSum(History &H,vector<float> *y_tminus1) const;
	void initWeights();
	void addPredictionWeight(Node *theNode);
	void setupHistoryWeights(int historySize);


	vector<float> historyWeights;
	vector<float> predictionWeights;
	vector<Node *> pWLookup;
//This is totally redundant and is only useful when loading things from file.  It might not even be set otherwise.
	vector<int> __pwIDs;
	
	float biasWeight;
};

#endif