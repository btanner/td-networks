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

#include "Node.h"
#include "Weights.h"
#include "OneStepNode.h"
#include <ostream>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include "convert.h"
#include "staticInits.h"
using std::cout;
using std::endl;

OneStepNode::OneStepNode(ifstream &i){
	i>>actCond>>obsToPredict>>globalIndex>>mcTargetTimeStep;
	i>>__parentID;
	int wSize;
	i>>wSize;
	
	for(int j=0;j<wSize;j++){
		theWeights.push_back(Weights());
		theWeights[j].Load(i);
	}
}

void OneStepNode::Save(ofstream &o) const{	
	o<<" "<<actCond<<" "<<obsToPredict<<" "<<globalIndex<<" "<<mcTargetTimeStep;
	if(parent==NULL)
		o<<" "<<-1;
	else
		o<<" "<<parent->globalIndex;
	
	o<<" "<<theWeights.size();
	for(int i=0;i<theWeights.size();i++)
		theWeights[i].Save(o);
	
}



float OneStepNode::inverseActivate(const float P) const{
	return P;
}

//*********************
//Learning code
//*********************
void OneStepNode::updateNode( float alphaDelta, const Input *theInput){
	int AOSelector=theInput->AOSelector;
	theWeights[AOSelector].updateWeights(alphaDelta, theInput);
}

//*********************
//Calculate predictions
//*********************
float OneStepNode::calcPrediction(History &H,vector<float> *y_tminus1) const{
	int AOSelector=H.getWeightsetIndex();
	float weightedSum=theWeights[AOSelector].weightedSum(H,y_tminus1);
	
	//New OO updates
	if(DET){printf("\tFinal Pred Weighted sum(%f) and activated %f\n",weightedSum,activate(weightedSum));}
//	printf("\tFinal Pred Weighted sum(%f) and activated %f\t using AO Selector %d\n",weightedSum,activate(weightedSum),AOSelector);

	return activate(weightedSum);
}

float OneStepNode::activate(float P) const{
	if(P>1.0f)
		return 1.0f;
	if(P<0.0f)
		return 0.0f;
	return P;
}

//*************************
//Boring functions below
//*************************
int OneStepNode::getObsToPredict() const{
	return obsToPredict;
}
int OneStepNode::getActCond() const{
	return actCond;
}

void OneStepNode::setParent(Node *p){
	this->parent=p;
}

Node *OneStepNode::getParent() const{
	return parent;
}

void OneStepNode::fixLinksAfterLoadFromFile(vector<Node *> &allNodes){
	for(int i=0;i<theWeights.size();i++)
		theWeights[i].fixLinksAfterLoadFromFile(allNodes);
}

string  OneStepNode::toString() const{
	string returnString;
	returnString="ID: "+stringify(globalIndex);
	if(parent==NULL)
		returnString+="\tParent: No parent";
	else
		returnString+="\tParent: "+stringify(parent->globalIndex)+"\t\t";

	returnString+="\t\tObsToPredict:\t "+stringify(obsToPredict);
	returnString+="\tLevel:\t"+stringify(mcTargetTimeStep);
	returnString+="\tActCond:\t"+stringify(actCond);
	returnString+="Weights: ";
	cout<<"About to this that loop..."<<endl;
	for(int i=0;i<theWeights.size();i++){
		cout<<"In that loop..."<<endl;
		returnString+=theWeights[0].toString()+"\n";
	}
	return returnString;
	
}

string  OneStepNode::toStringExtended() const{
	string returnString;
	returnString="Node: "+stringify(globalIndex);
	returnString+="\t Need to add weights to toStringExtended in node";
//	returnString+="\tBiasWeight: "+stringify(biasWeight);
	return returnString;
}

//This needs to be done very early,before weights are inited
void OneStepNode::createWeightsSets(History &H){
	for(int i=0;i<H.getWeightsetSize();i++)
		theWeights.push_back(Weights());	
}

void OneStepNode::initAllWeights(){
	for(int AOSelector=0;AOSelector<theWeights.size();AOSelector++)
		theWeights[AOSelector].initWeights();
}

void OneStepNode::addPredictionWeight(Node *theNode){
	for(int AOSelector=0;AOSelector<theWeights.size();AOSelector++)
		theWeights[AOSelector].addPredictionWeight(theNode);
}

void OneStepNode::setupHistoryWeights(int historySize){
	for(int AOSelector=0;AOSelector<theWeights.size();AOSelector++)
		theWeights[AOSelector].setupHistoryWeights(historySize);
}


//OneStepNode::OneStepNode(int obsToPredict){
//	OneStepNode(obsToPredict, Node::COND_ANY);
//}

//OneStepNode::OneStepNode( int obsToPredict, int actCond){
//	OneStepNode(obsToPredict, actCond, NULL, 0);
//}

/* Big Constructor, sets everything*/
OneStepNode::OneStepNode(int actCondValue, int ObsToPredictValue, Node *parentValue, int MCTimeStepValue):
actCond(actCondValue), 
obsToPredict(ObsToPredictValue), 
parent(parentValue),
mcTargetTimeStep(MCTimeStepValue){
}

