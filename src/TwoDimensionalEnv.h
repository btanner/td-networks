/*
 *  TwoDimensionalEnv.h
 *  TDNetsCPP
 *
 *  Created by Brian Tanner on 07/11/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef TWODIMENSIONALENV_H
#define TWODIMENSIONALENV_H

#include <string>
#include "TwoDimensionalMap.h"
using std::string;



#include "Environment.h"

#define DIR_UP 0
#define DIR_LEFT 3
#define DIR_RIGHT 1
#define DIR_DOWN 2




class TwoDimensionalEnv: public Environment{
public:
	TwoDimensionalEnv(TwoDimensionalMap &MAP);
	virtual string toString() const;

	
	virtual int getObservation(int Action);
	virtual void nextState(int Action,int Observation);
	virtual float getProbOfObservationGivenAction(int targetObs,int action);
	
	
private:
	int cRow;
	int cCol;
	int cDir;
	
	TwoDimensionalMap theMap;
	int currentState;
	int getObservationForCurrentState();
	int getObservationForState(int R, int C, int D);

};




#endif