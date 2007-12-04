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

#include<iostream>
#include <cmath>
#include "History.h"
#include "convert.h"

using std::cout;
using std::endl;

History::History(){
	//Careful... someone called the null History constructor
}

History::History(int actionCountVal, int obsCountVal,  char startParamVal, int featureLengthVal, int weightsetLengthVal):
	actionCount(actionCountVal), 
	obsCount(obsCountVal),
	featureLength(featureLengthVal),
	weightsetLength(weightsetLengthVal),
	startParam(startParamVal)
{
		//the length of a and o lists depend where we start counting
		if( startParam == 'a' ){
			aLengthFeature = (featureLength+1)/2;
			oLengthFeature = aLengthFeature-featureLength%2;
			aLengthWeightset = (weightsetLength+1)/2;
			oLengthWeightset = aLengthWeightset-weightsetLength%2;
		} 
		else {
			oLengthFeature = (featureLength+1)/2;
			aLengthFeature = oLengthFeature-featureLength%2;
			oLengthWeightset = (weightsetLength+1)/2;
			aLengthWeightset = oLengthWeightset-weightsetLength%2;
		}
		
		//calculate the lengths of the obsPart (for calculating indices)
		//and the totalFeatureSize (for getFeatureSize function)
		if(oLengthFeature>0) {
			obsPartSizeFeature=(int)(pow(obsCount,oLengthFeature));
			totalFeatureSize = obsPartSizeFeature*(int)(pow(actionCount,aLengthFeature));
		}
		else {
			obsPartSizeFeature=0;
			if(aLengthFeature>0)
				totalFeatureSize = (int)(pow(actionCount, aLengthFeature));
			else
				totalFeatureSize = 0;
		}
//Now do the same thing for the weightsets
		if(oLengthWeightset>0) {
			obsPartSizeWeightset=(int)(pow(obsCount,oLengthWeightset));
			totalWeightsetSize = obsPartSizeWeightset*(int)(pow(actionCount,aLengthWeightset));
		}
		else {
			obsPartSizeWeightset=0;
			if(aLengthWeightset>0)
				totalWeightsetSize = (int)(pow(actionCount, aLengthWeightset));
			else
				totalWeightsetSize = 1;
		}
		
/*old
		if(oLengthWeightset>0)
			obsPartSizeWeightset=(int)(pow(obsCount,oLengthWeightset));
		else
			obsPartSizeWeightset = 0;
		*/	
		//set aLength and oLength according to the largest value
		aLength = aLengthFeature;
		oLength = oLengthFeature;
		if( aLength < aLengthWeightset )
			aLength = aLengthWeightset;
		if( oLength < oLengthWeightset )
			oLength = oLengthWeightset;
		
		//make sure the length is valid
		if( aLength < 0 )
			aLength = 0;
		if( oLength < 0 )
			oLength = 0;
			
		//Give history index an initial value
		weightsetIndex=0;
		featureIndex = 0;
}

History::~History(){}

string  History::toString() const{
	string returnString;
	list<int>::const_iterator oit;
	list<int>::const_iterator ait;
	
	returnString="History: ";
	ait=aHist.begin();
	for(oit=oHist.begin(); oit!=oHist.end(); ++oit){
		returnString+=stringify(*oit)+stringify(*ait)+"\t";
		ait++;
	} 
		
	return returnString;
}

int History::updateAction(int action){
	int a = 0;
	if(aHist.size()>=aLength&&aHist.size()>0){
		a = aHist.back();
		aHist.pop_back();
	}
	aHist.push_front(action);
	setFeatureIndex(calcFeatureIndex());
	setWeightsetIndex(calcWeightsetIndex());
	return a;
}

void History::restoreAction(int action){
	aHist.pop_front();
	aHist.push_back(action);
}

int History::updateObservation(int observation){
	int o = 0;
	if(oHist.size()>=oLength&&oHist.size()>0){
		o = oHist.back();
		oHist.pop_back();
	}
	oHist.push_front(observation);
	setFeatureIndex(calcFeatureIndex());
	setWeightsetIndex(calcWeightsetIndex());
	return o;
}


void History::update(int action, int observation){
	if(aHist.size()>=aLength&&aHist.size()>0){
		aHist.pop_back();
	}
	if(oHist.size()>=oLength&&oHist.size()>0){
		oHist.pop_back();
	}
	aHist.push_front(action);
	oHist.push_front(observation);
//The overall history index
	setFeatureIndex(calcFeatureIndex());
//The aoIndex
	setWeightsetIndex(calcWeightsetIndex());
	
//	cout<<"action: "<<action<<" observation: "<<observation<<" new featureIndex: "<<getWeightsetIndex()<<endl;
}

//H is a temporal sequence (front most recent) of symbols
//size is the number of possible symbols
//length is how far back in time to consider

int History::getIndex(const list<int> &H, int size, int length) const{
	int HistIndex=0;
	int Mult=0;

	//New
	list<int>::const_iterator it;
	it=H.begin();
	
	for(int i=0; i<length && it!=H.end(); ++it, ++i){
		int j = (int)(pow(size,Mult));
		int k = *it;
		HistIndex+=k*j;
		Mult++;	
	} 
	return HistIndex;
}

	

