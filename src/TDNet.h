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


#ifndef TDNET_H
#define TDNET_H

#include "Node.h"
#include "Printable.h"
#include "NodeFactory.h"
#include "Serializable.h"
#include <vector>

using std::vector;

class History;

class TDNet : public Printable, public Serializable {
public:
	
	//Contracted interface implementations
	virtual string toString() const;
	virtual void Save(ofstream &o) const;

	
	TDNet(NodeFactory F,int depth=1, int actionCount=1, int obsCount=2);
	TDNet();
	TDNet(NodeFactory F, ifstream &i,vector<Node *> &allNodes);
	virtual ~TDNet();
	void addToGlobalVector(vector<Node *> &allNodes);
	void hookNodesToEachOther();
	void initAllWeights();
	void createWeightsSets(History &H);
	int getNodeCount() const;
	int getActionCount() const;
	int getObsCount() const;

	vector<Node *> Nodes;

		
private:		
	int depth;
	int actionCount;
	int obsCount;
	NodeFactory theNodeFactory;
	void HangNetworkOff(Node *parent,int Level);
	
};
inline int TDNet::getActionCount() const{
	return actionCount;	
}
inline int TDNet::getObsCount() const{
	return obsCount;	
}

inline TDNet::TDNet(){
//Careful... someone called the null TDNEt constructor 
}
inline int TDNet::getNodeCount() const{
	return Nodes.size();
}
#endif
