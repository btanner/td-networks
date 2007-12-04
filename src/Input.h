/*
 *  Input.h
 *  TDNetsCPP
 *
 *  Created by Brian Tanner on 07/09/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef INPUT_H
#define INPUT_H

#include <vector>
#include "History.h"

using std::vector;

class Input{
public:
	static int timesInited;
	int timeStep;
	int historyNumber;
	int AOSelector;
	vector<float> *yt;
	Input(int timeStepValue, History const &H, vector<float> *predictions);
//Used now that we are basically making dummy inputs
	Input();
	Input *set(int timeStepValue, History const &H, vector<float> *predictions);
};
	
#endif