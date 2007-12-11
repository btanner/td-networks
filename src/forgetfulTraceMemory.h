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
	int totalSize;
	int head;
	int tail;
	int currentSize;
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
