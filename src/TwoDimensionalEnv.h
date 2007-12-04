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