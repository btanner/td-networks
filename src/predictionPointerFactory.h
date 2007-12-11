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
