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

