/*
 *  SolutionMethod.h
 *  TDNetsCPP
 *
 *  Created by Brian Tanner on 07/11/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef SOLUTIONMETHOD_H
#define SOLUTIONMETHOD_H

#include <string>
#include "Serializable.h"

using std::string;

class SolutionMethod:public Serializable{
public:
	enum SolutionType
{
	TDNetAlgSolType,
	Other
};

	SolutionMethod(SolutionType theType, float stepSizeVal);
	~SolutionMethod();
	
	virtual void updateState(int action, int observation)=0;
	virtual float getPForAO(int action, int observation)const=0;
	
	virtual void saveToFile(string fileName) const;
	static SolutionMethod* loadFromFile(string fileName) ;
	float getStepSize() const;
	void setStepSize(float stepsize);


	SolutionType theSolutionType;

protected:
	float stepSize;


};

#endif