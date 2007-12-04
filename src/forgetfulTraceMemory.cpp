/*
 *  forgetfulTraceMemory.cpp
 *  TDNetsCPP
 *
 *  Created by Brian Tanner on 21/09/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */

#include "forgetfulTraceMemory.h"

void forgetfulTraceMemory::setAtEnd(TraceNode *oldNode,float curVal){
	tracePointers[tail]->set(oldNode,curVal);
	tail++;
	if(tail==totalSize)
		tail=0;
	currentSize++;
	if(tail==head){
		exit(101);
	}
}

void forgetfulTraceMemory::setAtEnd(Node *theNodeVal, Node* currentTargetVal, int tVal, float lastValueVal, float origValueVal, float alphaVal){
	tracePointers[tail]->set(theNodeVal, currentTargetVal, tVal, lastValueVal, origValueVal, alphaVal);
	tail++;
	if(tail==totalSize)
		tail=0;
	currentSize++;
	if(tail==head){
		exit(101);
	}
	
}


TraceNode *forgetfulTraceMemory::popFront(){
	TraceNode *returnVal=tracePointers[head];
	head++;
	if(head==totalSize)
		head=0;
	currentSize--;
	return returnVal;
}


forgetfulTraceMemory::forgetfulTraceMemory(int totalCount):
totalSize(totalCount), head(0), tail(0),currentSize(0){
	for(int i=0;i<totalSize;i++){
		tracePointers.push_back(new TraceNode());
	}
}

