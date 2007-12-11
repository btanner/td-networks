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
