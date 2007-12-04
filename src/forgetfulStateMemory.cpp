/*
 *  forgetfulStateMemory.cpp
 *  TDNetsCPP
 *
 *  Created by Brian Tanner on 21/09/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */

#include "forgetfulStateMemory.h"


void forgetfulStateMemory::addToEnd(Input *i){
	tail++;
	if(tail==head){
		exit(100);
	}
	if(tail>=totalSize)
		tail=0;
	inputPointers[tail]=i;
	currentSize++;
}

void forgetfulStateMemory::popFront(){
	head++;
	if(head>=totalSize)
		head=0;
	currentSize--;
}


Input *forgetfulStateMemory::getRelativeToEnd(int loc) const{
	int theSpot=tail-loc;
	if(theSpot<0)
		theSpot=totalSize+theSpot;
	return inputPointers[theSpot];
}


forgetfulStateMemory::forgetfulStateMemory(int totalCount):
totalSize(totalCount), head(0), tail(0),currentSize(0){
	for(int i=0;i<totalSize;i++){
		inputPointers.push_back(NULL);
	}
}
