/*
 *  TraceNode.h
 *  TDNetsCPP
 *
 *  Created by Brian Tanner on 07/09/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef TRACENODE_H
#define TRACENODE_H

#include <vector>
#include "Node.h"
#include "Input.h"
#include "math.h"
using std::vector;


class lambdaDetails{
public:
	float lambdas[10];
	lambdaDetails(float l);
	float getLambda(const int &power) const;	
};

inline float lambdaDetails::getLambda(const int &power) const{
	return lambdas[power];
}



class TraceNode {
public:
	Node *theNode;
	Node *currentTarget;
	
//Inlined
	void ageParent();
	//Inlined
	bool checkCondition(int a) const;
	int t;
	float lastValue;
	float origValue;
	TraceNode(Node *theNodeVal, Node* currentTargetVal, int tVal, float lastValueVal, float origValueVal, float alphaVal);
	TraceNode(TraceNode &oldNode, float curVal);
//Useful because we are using a forgetfultracememory now
	TraceNode();
	TraceNode *set(Node *theNodeVal, Node* currentTargetVal, int tVal, float lastValueVal, float origValueVal, float alphaVal);
	TraceNode *set(TraceNode *oldNode, float curVal);
	float alpha;
	
	void learn(const int lambdaPower,const Input *theInput,const vector<float> *y,const int O) const;
	void cheatLearn(const int lambdaPower,const Input *theInput,const vector<float> *y,const int O, float probAct0, float probAct1) const;
	
//	static float lambda;

};
inline void TraceNode::ageParent(){
	currentTarget=currentTarget->getParent();
}

inline bool TraceNode::checkCondition(int a) const{
	return(currentTarget->getActCond()==-1|currentTarget->getActCond()==a);
}











#endif