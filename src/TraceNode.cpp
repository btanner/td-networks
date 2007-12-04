/*
 *  TraceNode.cpp
 *  TDNetsCPP
 *
 *  Created by Brian Tanner on 07/09/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */

#include "TraceNode.h"
#include <iostream>
#include <stdio.h>
#include <math.h>
#include "staticInits.h"
using std::cout;
using std::endl;


lambdaDetails::lambdaDetails(float l){
	for(int i=0;i<10;i++){
		lambdas[i]=powf(l, i);
	}
}


void TraceNode::cheatLearn(const int lambdaPower,const Input *theInput,const vector<float> *yt,const int O, float probAct0, float probAct1) const{
	float P=lastValue;
	float Z;
	//	static lambdaDetails lambda=lambdaDetails(this->lambda);
	
	if(DET){printf("\t\t\tDoing learning on node %d",theNode->globalIndex);}
	if(currentTarget==NULL){
//		cout<<"ObstoP: "<<theNode->getObsToPredict()<<" ActToP: "<<theNode->getActCond()<<" Observation was: "<<O<<" - the target I'm using is: ";
		if(theNode->getActCond()==0)
			Z=probAct1;
		else
			Z=probAct0;
		
//		cout<<Z;
		if(theNode->getObsToPredict()!=O){
			Z=1.0f-Z;
//			cout<<" which was reversed to: "<<Z;
		}
//		cout<<endl;
	}else{
		Z=(*yt)[currentTarget->globalIndex];
		if(DET){printf("PTarget %f", (*yt)[currentTarget->globalIndex]);}
		
	}
	
	float error=Z-P;
	if(DET){printf(", error is %f",error);
		printf("\t  target %f\t p=%f so error is %f\n",Z,P,error);}
	
	//New (slower!?)
	//	float lambdaWeight=lambda.getLambda(lambdaPower);
	//Old (faster?!?)
	float lambdaWeight=powf(LAMBDA,lambdaPower);
	//		cout<<"Old way gives: "<<this->lambda<<" ^ "<<lambdaPower<<" = "<<powf(this->lambda,lambdaPower)<<endl;
	//		cout<<"New way returns: "<<lambdaWeight<<endl;
	
	float dP=theNode->inverseActivate(P);
	//	cout<<"activate of 0 is : "<<theNode->activate(0.0f)<<endl;
	//	cout<<"Inverse activate of .5 is : "<<theNode->inverseActivate(0.5f)<<endl;
	
	float delta=error*dP*lambdaWeight;
	theNode->updateNode(delta*alpha, theInput);
}

void TraceNode::learn(const int lambdaPower,const Input *theInput,const vector<float> *yt,const int O) const{
	float P=lastValue;
	float Z;
//	static lambdaDetails lambda=lambdaDetails(this->lambda);
	
	if(DET){printf("\t\t\tDoing learning on node %d",theNode->globalIndex);}

//	cout<<"Alpha is: "<<alpha<<endl;
	if(currentTarget==NULL){
		if(theNode->getObsToPredict()==O){
			if(DET){printf("ObsTarget, Match");}
			Z=1.0f;
		}
		else{
			Z=0.0f;
			if(DET){printf("ObsTarget, NOMatch");}

		}
	}else{
		Z=(*yt)[currentTarget->globalIndex];
		if(DET){printf("PTarget %f", (*yt)[currentTarget->globalIndex]);}

	}
	
	float error=Z-P;
	if(DET){printf(", error is %f",error);
		printf("\t  target %f\t p=%f so error is %f\n",Z,P,error);}

	//New (slower!?)
	//	float lambdaWeight=lambda.getLambda(lambdaPower);
	//Old (faster?!?)
	float lambdaWeight=powf(LAMBDA,lambdaPower);
	//		cout<<"Old way gives: "<<this->lambda<<" ^ "<<lambdaPower<<" = "<<powf(this->lambda,lambdaPower)<<endl;
	//		cout<<"New way returns: "<<lambdaWeight<<endl;

	float dP=theNode->inverseActivate(P);
//	cout<<"activate of 0 is : "<<theNode->activate(0.0f)<<endl;
//	cout<<"Inverse activate of .5 is : "<<theNode->inverseActivate(0.5f)<<endl;
	
	float delta=error*dP*lambdaWeight;
//	cout<<"Learning Node "<<theNode->globalIndex<<" with AOSelector: "<<theInput->AOSelector<<"\t\t The prediction we're updating was: "<<P<<endl;
	theNode->updateNode(delta*alpha, theInput);
}

TraceNode *TraceNode::set(TraceNode *oldNode, float curVal){
	theNode=oldNode->theNode;
	currentTarget=oldNode->currentTarget;
	t=oldNode->t;
	lastValue=curVal;
	origValue=oldNode->origValue;
	alpha=oldNode->alpha;
	return this;
}
TraceNode *TraceNode::set(Node *theNodeVal, Node* currentTargetVal, int tVal, float lastValueVal, float origValueVal, float alphaVal){
	alpha=alphaVal;
	theNode=theNodeVal;
	currentTarget=currentTargetVal;
	t=tVal;
	lastValue=lastValueVal;
	origValue=origValueVal;
	return this;
}

TraceNode::TraceNode(){}

TraceNode::TraceNode(Node *theNodeVal, Node* currentTargetVal, int tVal, float lastValueVal, float origValueVal, float alphaVal)
:	theNode(theNodeVal),
currentTarget(currentTargetVal),
t(tVal),
lastValue(lastValueVal),
origValue(origValueVal),
alpha(alphaVal){}

TraceNode::TraceNode(TraceNode &oldNode, float curVal){
	theNode=oldNode.theNode;
	//Aging happens elsewhere, after the action condition check
	currentTarget=oldNode.currentTarget;
	t=oldNode.t;
	lastValue=curVal;
	origValue=oldNode.origValue;
	alpha=oldNode.alpha;
}
