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

#include "Weights.h"

#include <vector>
#include <iostream>
#include "Node.h"
#include "convert.h"
#include <stdio.h>
#include "staticInits.h"
#include <string>


using std::string;

using std::vector;
using std::cout;
using std::endl;


void Weights::updateWeights(float alphaDelta, const Input *theInput){
	//Update the bias term
	int totalWeights=2+predictionWeights.size();

	alphaDelta*=1.0f/(float)totalWeights;
	biasWeight+=alphaDelta;

	//Then do the history	
	historyWeights[theInput->historyNumber]+=alphaDelta;

	for(unsigned int i=0;i<predictionWeights.size();i++){
		predictionWeights[i]+=alphaDelta*(*theInput->yt)[pWLookup[i]->globalIndex];
	}
}

float Weights::weightedSum(History &H,vector<float> *y_tminus1) const{

	float historyPart=0;
	float predPart=0;
	float biasPart=0;
	
	biasPart=biasWeight;
	if(DET){	printf("\t\t: Bias Weight: %f PWeights: ",biasWeight);}

	if(DET){	printf("H%d: %f\t",H.getFeatureIndex(), historyWeights[H.getFeatureIndex()]);}

	historyPart=historyWeights[H.getFeatureIndex()];
	
	int pCount=predictionWeights.size();
	for(int i=0;i<pCount;i++){
		predPart+=predictionWeights[i]*(*y_tminus1)[pWLookup[i]->globalIndex];
	}
	
	//	cout<<"biasPart: "<<biasPart<<" historyPart: "<<historyPart<<" predPart: "<<predPart<<endl;
	float total=biasPart+historyPart+predPart;
	return total;
}

void Weights::initWeights(){
	int totalWeights=2+predictionWeights.size();
	float initWeight=0.0f/((float)(totalWeights));
	biasWeight=initWeight;
	for(unsigned int i=0;i<historyWeights.size();i++)
		historyWeights[i]=initWeight;
	for(unsigned int i=0;i<predictionWeights.size();i++)
		predictionWeights[i]=initWeight;
}
Weights::Weights(){}

Weights::~Weights(){}

void Weights::addPredictionWeight(Node *theNode){
	predictionWeights.push_back(.1f);
	pWLookup.push_back(theNode);	
}
/*I think this just puts them in, I believe they are overwritten */
void Weights::setupHistoryWeights(int historySize){
	for(int i=0;i<historySize;i++)
		historyWeights.push_back(0.1f);
	biasWeight=0.1f;
	
}


void Weights::Save(ofstream &o) const{	
	o<<" "<<historyWeights.size();
	for(unsigned int i=0;i<historyWeights.size();i++)
		o<<" "<<historyWeights[i];
	o<<" "<<predictionWeights.size();
	for(unsigned int i=0;i<predictionWeights.size();i++)
		o<<" "<<predictionWeights[i];
	o<<" "<<biasWeight;
	o<<" "<<pWLookup.size();
	for(unsigned int i=0;i<pWLookup.size();i++)
		o<<" "<<pWLookup[i]->globalIndex;
	
		
}
void Weights::Load(ifstream &i) {
	int hSize;int pSize;int pWSize;
	i>>hSize;
	for(int j=0;j<hSize;j++){
		float tmp;
		i>>tmp;
		historyWeights.push_back(tmp);
	}
	i>>pSize;
	for(int j=0;j<pSize;j++){
		float tmp;
		i>>tmp;
		predictionWeights.push_back(tmp);
	}
	i>>biasWeight;
	i>>pWSize;
	for(int j=0;j<pWSize;j++){
		int tmp;
		i>>tmp;
		__pwIDs.push_back(tmp);
		pWLookup.push_back(NULL);//place holder
	}
}

void Weights::fixLinksAfterLoadFromFile(vector<Node *> &allNodes){
	for(unsigned int i=0;i<__pwIDs.size();i++){
		pWLookup[i]=allNodes[__pwIDs[i]];
	}
}


string  Weights::toString() const{
	string returnString;
	returnString="dummy for now";
	//for(int i=0;i<predictionWeights.size();i++)
	//returnString+="connection of: "+stringify(predictionWeights[i])+" to node: "+stringify(pWLookup[i]->globalIndex)+"\n";
	return returnString;
}
