/*
 *  TDNet.cpp
 *  TDNetsCPP
 *
 *  Created by Brian Tanner on 30/08/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */

#include "TDNet.h"
#include <vector>
#include <iostream>
#include "convert.h"
#include "OneStepNode.h"
#include "SigmoidOneStepNode.h"
#include "CrossEntropyOneStepNode.h"
#include "staticInits.h"

using std::vector;
using std::cout;
using std::endl;


void TDNet::Save(ofstream &o) const{
	if (DET)
		cout<<"Serializing TD Network Structure... all "<<Nodes.size()<<" nodes!"<<endl;
	o<<" "<<depth<<" "<<actionCount<<" "<<obsCount;
	o<<" "<<Nodes.size();
	for(int i=0;i<Nodes.size();i++)
		Nodes[i]->Save(o);	
}

void TDNet::addToGlobalVector(vector<Node *> &allNodes){
	for(int i=0;i<Nodes.size();i++){
		allNodes.push_back(Nodes[i]);
		Nodes[i]->globalIndex=allNodes.size()-1;
	}	
}

TDNet::TDNet(NodeFactory F, ifstream &i,vector<Node *> &allNodes){
	theNodeFactory=F;
	i>>depth>>actionCount>>obsCount;

	int nSize;
	i>>nSize;
	
	allNodes.resize(nSize);
	//Ok, put all nodes and their weights back, and they are in the global place as well
	for(int j=0;j<nSize;j++){
		Nodes.push_back(F.createNode(i));
		allNodes[Nodes[j]->globalIndex]=Nodes[j];
	}
	//Now iterate, connecting the parents, and fixing the weights
	for(int j=0;j<nSize;j++){
		Nodes[j]->fixLinksAfterLoadFromFile(allNodes);
		if(Nodes[j]->__parentID==-1)
			Nodes[j]->setParent(NULL);
		else
			Nodes[j]->setParent(allNodes[Nodes[j]->__parentID]);
	}
}




TDNet::TDNet( NodeFactory F,int depthValue, int actionCountValue, int obsCountValue) :
depth(depthValue), actionCount(actionCountValue), obsCount(obsCountValue), theNodeFactory(F){
	for (int j=0; j<actionCount; j++) {
		for (int i=1; i<obsCount; i++) {
			Nodes.push_back(theNodeFactory.createNode(j,i,NULL,0));
		}
	}
	if(depth>1){
		int nodesSoFar=Nodes.size();
		for (int k=0;k<nodesSoFar; k++) {
			HangNetworkOff(Nodes[k],1);
		}
	}
}
void TDNet::HangNetworkOff(Node *parent,int Level){
	for (int j=0; j<actionCount; j++) {
		int obsToPredict=parent->getObsToPredict();
		Nodes.push_back(theNodeFactory.createNode(j,obsToPredict,parent,Level));
			
		if(Level+1<depth)
			HangNetworkOff(Nodes[Nodes.size()-1],Level+1);
	}
}

string  TDNet::toString() const{
	string returnString;
	returnString="TDNet Printout: \n";
	
	for (int i=0; i<Nodes.size(); i++) {
		returnString+="Node ";
		returnString+=stringify(i);
		returnString+=":\t";
		returnString+=Nodes[i]->toString();
		returnString+="\n";
	}
	return returnString;
}




void TDNet::initAllWeights(){
	for(int i=0;i<Nodes.size();i++){
		Nodes[i]->initAllWeights();
	}	
}

void TDNet::createWeightsSets(History &H){
	for(int i=0;i<Nodes.size();i++){
		Nodes[i]->createWeightsSets(H);
	}		
}





void TDNet::hookNodesToEachOther(){
	for(int i=0;i<Nodes.size();i++){
		for(int j=0;j<Nodes.size();j++){
			Nodes[i]->addPredictionWeight(Nodes[j]);
		}		
	}
}
