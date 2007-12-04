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

#include "forgetfulStateMemory.h"


void forgetfulStateMemory::addToEnd(Input *i){
	tail++;
	if(tail==head){
		exit(100);
	}
	if(tail>=totalSize)
		tail=0;
	inputPointers[tail]=i;
	currentSize++;
}

void forgetfulStateMemory::popFront(){
	head++;
	if(head>=totalSize)
		head=0;
	currentSize--;
}


Input *forgetfulStateMemory::getRelativeToEnd(int loc) const{
	int theSpot=tail-loc;
	if(theSpot<0)
		theSpot=totalSize+theSpot;
	return inputPointers[theSpot];
}


forgetfulStateMemory::forgetfulStateMemory(int totalCount):
totalSize(totalCount), head(0), tail(0),currentSize(0){
	for(int i=0;i<totalSize;i++){
		inputPointers.push_back(NULL);
	}
}
