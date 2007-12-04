/*
 *  Input.cpp
 *  TDNetsCPP
 *
 *  Created by Brian Tanner on 07/09/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */

#include "Input.h"
#include <vector>
#include <iostream>

using std::vector;
using std::cout;
using std::endl;

Input::Input(){
	timesInited++;
}

//Added because we are not making these on the fly, but getting them from a factory
Input *Input::set(int timeStepValue, History const &H, vector<float> *predictions){
	timeStep=timeStepValue;
	historyNumber=H.getFeatureIndex();
	AOSelector=H.getWeightsetIndex();
	yt=predictions;
	return this;
}

Input::Input(int timeStepValue, History const &H, vector<float> *predictions) 
:	timeStep(timeStepValue), /*new code*/ yt(predictions),
	historyNumber(H.getFeatureIndex()), AOSelector(H.getWeightsetIndex()){
		vector<float>::const_iterator it;
		
		timesInited++;
		if(historyNumber>1000||historyNumber<0)
			cout<<"\t\tInput initialized with value: "<<historyNumber<<endl;
		//Now we are using pointers to vectors of floats instead of vectors of floats, should be able to be cheaper (did it in the top part)
		//for(it=predictions->begin(); it!=predictions->end();it++){
		//	yt.push_back(*it);
		//}
		
			
}
int Input::timesInited=0;

