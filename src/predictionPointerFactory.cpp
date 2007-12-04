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


