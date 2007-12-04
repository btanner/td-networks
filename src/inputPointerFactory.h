/*
 *  inputPointerFactory.h
 *  TDNetsCPP
 *
 *  Created by Brian Tanner on 21/09/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef INPUTPOINTERFACTORY_H
#define INPUTPOINTERFACTORY_H
#include "Input.h"
#include <vector>

using std::vector;

class inputPointerFactory{
public:
	inputPointerFactory(int totalCount=100);	
//Inlined
	Input *getNextFreePointer();
private:
//Inlined
	void updatePointer();	
	
	//Want a vector of pointers to Inputs
	vector<Input *> inputPointers;
	int totalSize;
	int nextFree;
};
inline Input *inputPointerFactory::getNextFreePointer(){
	Input *returnVal=inputPointers[nextFree];
	updatePointer();
	return returnVal;
}

inline void inputPointerFactory::updatePointer(){
	nextFree++;
	if(nextFree>=totalSize)
		nextFree=0;
}


#endif