/*
 *  inputPointerFactory.cpp
 *  TDNetsCPP
 *
 *  Created by Brian Tanner on 21/09/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */

#include "inputPointerFactory.h"

inputPointerFactory::inputPointerFactory(int totalCount):
totalSize(totalCount), nextFree(0){
	for(int i=0;i<totalSize;i++){
		inputPointers.push_back(new Input());
	}
}


