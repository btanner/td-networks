/*
 *  HistoryAlgorithm.h
 *  HistorysCPP
 *
 *  Created by Brian Tanner on 07/11/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef HISTALGORITHM_H
#define HISTALGORITHM_H

#include "SolutionMethod.h"
#include "History.h"
#include <iostream>
using namespace::std;

class HistoryAlgorithm: public SolutionMethod{
public:
	
	
	HistoryAlgorithm(int aCount, int oCount, int hLength);
	HistoryAlgorithm(ifstream &i);
	~HistoryAlgorithm();
	
	virtual void Save(ofstream &o) const;
	
	virtual void updateState(int action, int observation);
	virtual float getPForAO(int action, int observation) const;

protected:
	/*Parameters we'll need to save.*/
	int historyLength;
	int actionCount;
	int observationCount;
	float *numH;
	float *numOH;
	
	private:
	/*Parameters we'll need to recreate*/
	int t;
//	float alpha;

	//Create the history structure
	History H;
};
#endif