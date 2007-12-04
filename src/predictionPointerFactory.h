/*
 *  predictionPointerFactory.h
 *  TDNetsCPP
 *
 *  Created by Brian Tanner on 20/09/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef PREDICTIONPOINTERFACTORY_H
#define PREDICTIONPOINTERFACTORY_H
#include <vector>

using std::vector;

class predictionPointerFactory{
public:
	
//Inlined functions
	predictionPointerFactory();
	predictionPointerFactory(int eachSize, int totalCount=100);	
	vector<float> *getNextFreePointer();
private:
	void updatePointer();	
		
	//Want a vector of pointers to vectors of floats
	vector<vector<float> *> predictionPointers;
	int totalSize;
	int nextFree;
};
inline vector<float> *predictionPointerFactory::getNextFreePointer(){
	vector<float > *returnVal=predictionPointers[nextFree];
	updatePointer();
	return returnVal;
}

inline void predictionPointerFactory::updatePointer(){
	nextFree++;
	if(nextFree>=totalSize)
		nextFree=0;
}


#endif