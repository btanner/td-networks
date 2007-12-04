/*
 *  predictionPointerFactory.cpp
 *  TDNetsCPP
 *
 *  Created by Brian Tanner on 21/09/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */

#include "predictionPointerFactory.h"

#include <iostream>

using std::cout;
using std::endl;

predictionPointerFactory::predictionPointerFactory(){
	//Careful... someone called the null predictionPointerFactory 
}

predictionPointerFactory::predictionPointerFactory(int eachSize, int totalCount):
	totalSize(totalCount), nextFree(0)
{
	
	for(int i=0;i<totalSize;i++){
		predictionPointers.push_back(new vector<float>());
		for(int j=0;j<eachSize;j++)
			predictionPointers[i]->push_back(0.0f);
	}
}


