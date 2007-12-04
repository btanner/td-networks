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

#ifndef ENVIRONMENT_H_BRIAN
#define ENVIRONMENT_H_BRIAN
#include "Printable.h"

using std::string;

class Environment : public Printable{
public:
	Environment(int observationCount, int actionCount);
	~Environment();

	virtual int getObservation(int Action)=0;
	virtual void nextState(int Action,int Observation)=0;

	//For error checking
	virtual float getProbOfObservationGivenAction(int targetObs,int action)=0;
	
	//Inlined
	int getActionCount();
	int getObservationCount();
	//Choose a random action
	int nextAction();
	
	
	
protected:
	int actionCount;
	int observationCount;
};


inline int Environment::nextAction(){
	int nextAction;
	nextAction = (int) ((double)getActionCount() * (rand() / (RAND_MAX + 1.0)));
	return nextAction;
}


inline int Environment::getActionCount(){
	return actionCount;
}

inline int Environment::getObservationCount(){
	return observationCount;
}
#endif
