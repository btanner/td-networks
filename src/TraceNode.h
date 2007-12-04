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