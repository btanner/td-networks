/*
 *  forgetfulTraceMemory.h
 *  TDNetsCPP
 *
 *  Created by Brian Tanner on 21/09/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef FORGETFULTRACEMEMORY_H
#define FORGETFULTRACEMEMORY_H
#include <vector>
#include "TraceNode.h"

using std::vector;

//Need to be able to add to the end
class forgetfulTraceMemory{
public:
	forgetfulTraceMemory(int totalCount=100);
	
	void setAtEnd(TraceNode *oldNode, float curVal);
	void setAtEnd(Node *theNodeVal, Node* currentTargetVal, int tVal, float lastValueVal, float origValueVal, float alphaVal);

	TraceNode *popFront();
	
	//Inlined functions
	int size() const;
	void clear();
	
private:
		
	vector<TraceNode *> tracePointers;
	int currentSize;
	int totalSize;
	int head;
	int tail;
};

inline int forgetfulTraceMemory::size() const {
	return currentSize;
}

inline void forgetfulTraceMemory::clear(){
	head=0;
	tail=0;
	currentSize=0;
}

#endif

