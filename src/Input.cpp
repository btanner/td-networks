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

