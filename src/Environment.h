/*
 *  Environment.h
 *  TDNetsCPP
 *
 *  Created by Brian Tanner on 07/11/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */

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
