/*
 *  TDNet.h
 *  TDNetsCPP
 *
 *  Created by Brian Tanner on 30/08/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */
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