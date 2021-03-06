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

#ifndef POMDPENV_H
#define POMDPENV_H
#include <vector>
#include <string>
#include "Environment.h"

using std::vector;
using std::string;

class pomdpEnv : public Environment{
public:
	pomdpEnv(int stateCount,int observationCount, int actionCount);
	virtual ~pomdpEnv();
	virtual string toString() const;
	virtual int getObservation(int Action);

	//Inlined
	virtual void nextState(int Action,int Observation);
		int getStateCount();

	//For error checking
	float getProbOfObservationGivenAction(int targetObs,int action);
	float getProbOfObservationGivenActions(int targetObservation,int a1, int a2);
	float* getBeliefState();
	void checkConsistency();

protected:
//Inlined
	int calcIndex(int a, int s1, int s2) const;
	int calcIndex(int s, int o) const;
	float* transitions; //[a][s][s]
	float* observations; //[s][o]
	float* belief; //[s]
	float* beliefTemp; //[s]
	float* beliefTemp1; //[s]
	float* oSample; // [o]
	int stateCount;
	void setTransition(int action,int S1,int S2,float Value);
	void setObservation(int S1,int Obs, float Value);
	float getTransition(int a, int s1, int s2) const;
	float getObservation(int S1,int Obs);

	
private:
	void updateBelief(int action, int observation);
	int sampleObservation(int action);
};

inline float pomdpEnv::getObservation(int S1,int Obs){
	return observations[calcIndex(S1,Obs)];
}

inline 	void pomdpEnv::setObservation(int S1,int Obs, float Value){
	if(calcIndex(S1,Obs)>getObservationCount()*stateCount||calcIndex(S1,Obs)<0)
		printf("Tring to possibly set an invalid observation %d\n",calcIndex(S1,Obs));
	observations[calcIndex(S1,Obs)]=Value;
}

inline void pomdpEnv::setTransition(int action,int S1,int S2,float Value){
	if(calcIndex(action,S1,S2)>stateCount*getActionCount()*stateCount||calcIndex(action,S1,S2)<0)
		printf("Gonna go set transition[%d] to %f - the max index is %d... a=%d S1=%d S2=%d\n",calcIndex(action,S1,S2),Value,stateCount*actionCount*stateCount,action,S1,S2);

	transitions[calcIndex(action,S1,S2)]=Value;	
}


inline int pomdpEnv::getStateCount(){
	return stateCount;
}


inline void pomdpEnv::nextState(int Action,int Observation){
	updateBelief(Action,Observation);
}

inline int pomdpEnv::getObservation(int Action){
	return sampleObservation(Action);
}


/*Unsure if this is correct */
/* Adapted from http://webster.cs.ucr.edu/AoA/Windows/HTML/Arraysa2.html */
inline int pomdpEnv::calcIndex(int a, int s1, int s2) const{
	int index=0;
	//Big Dimension
	int aSize=a*stateCount*stateCount;
	//Second Dimension
	int sSize=s1*stateCount;
	//Third
	int s1Size=s2;
	index=aSize+sSize+s1Size;
	return index;
}

/* Also unsure if this is correct */
inline int pomdpEnv::calcIndex(int s, int o) const{
	return(o*stateCount+s);
}

inline float pomdpEnv::getTransition(int a, int s1, int s2) const{
	return(transitions[calcIndex(a,s1,s2)]);
}


#endif
