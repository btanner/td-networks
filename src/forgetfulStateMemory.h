/*
 *  forgetfulStateMemory.h
 *  TDNetsCPP
 *
 *  Created by Brian Tanner on 21/09/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef FORGETFULSTATEMEMORY_H
#define FORGETFULSTATEMEMORY_H
#include <vector>
#include "Input.h"

using std::vector;

//Need to be able to add to the end
class forgetfulStateMemory{
public:
	forgetfulStateMemory(int totalCount=100);

	void addToEnd(Input *I);
	void popFront();
	Input *getRelativeToEnd(int loc) const;

//Inlined functions
	int size() const;
	
private:

	vector<Input *> inputPointers;
	int currentSize;
	int totalSize;
	int head;
	int tail;
};

inline int forgetfulStateMemory::size() const {
	return currentSize;
}

#endif

