/*
 *  history.h
 *  TDNetsCPP
 *
 *  Created by Brian Tanner on 23/08/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef HISTORY_H
#define HISTORY_H

#include <string>
#include <list>
#include "Printable.h"
#include "staticInits.h"
#include <iostream>

using std::string;
using std::list;
using std::cout;
using std::endl;

class History : public Printable {
public:
	//probably a stupid name for the "start with a" or "start with o"
	History(int actionCount, int obsCount, char startParam='a', int featureLength=1, int weightsetLength=1);
		History();
	~History();
	virtual string toString() const;
	void update(int action, int observation); //
	int updateAction(int action); //returns the popped-off action, if any
	void restoreAction(int action); //replaced a popped-off action to the end of the list
	int updateObservation(int observation); //returns the popped-off observation, if any

//Inlined
	int getFeatureIndex() const;
	int getWeightsetIndex() const;
	int getIndex(const list<int> &H, int size, int l) const;
	int getFeatureSize() const;
	int getWeightsetSize() const;
	
private:
	list<int> aHist;
	list<int> oHist;

//Inlined
	void setWeightsetIndex(int newIndex);
	void setFeatureIndex(int newIndex);
	int calcWeightsetIndex() const;
	int calcFeatureIndex() const;
	
	char startParam;
	
	int featureIndex;
	int weightsetIndex;

	int actionCount;
	int obsCount;
	int featureLength;
	int weightsetLength;

	int aLength;
	int aLengthFeature;
	int aLengthWeightset;
	int oLength;
	int oLengthFeature;
	int oLengthWeightset;
	
	int obsPartSizeFeature;
	int obsPartSizeWeightset;
	int totalFeatureSize; //this is the only one so far needed outside the code
	int totalWeightsetSize; //this is the only one so far needed outside the code
//	int actPartSize;
//	int totalSize;
};

inline int History::getWeightsetIndex() const{
	return weightsetIndex;
}


inline int History::getFeatureIndex() const{
	//sanity check necessary?
	return featureIndex;
}

inline int History::getWeightsetSize() const{
	return totalWeightsetSize;
}
inline int History::getFeatureSize() const{
       return totalFeatureSize;
}


inline void History::setWeightsetIndex(int val){
	weightsetIndex=val;
}

inline void History::setFeatureIndex(int val){
	featureIndex=val;
}

inline int History::calcFeatureIndex() const {
	int obsIndex;
	int actIndex;
	int hIndex;
	
	obsIndex=getIndex(oHist,obsCount,oLengthFeature);
	actIndex=getIndex(aHist,actionCount,aLengthFeature);

	if(obsPartSizeFeature>0)
		hIndex=obsPartSizeFeature*actIndex+obsIndex;
	else
		hIndex=actIndex;
	//cout<<"hIndex ("<<hIndex<<") calculated as: "<<oLengthFeature<<" * "<<actIndex<<"+"<<obsIndex<<endl;
	return hIndex;
}

inline int History::calcWeightsetIndex() const {
	int obsIndex;
	int actIndex;
	int hIndex;
	
	obsIndex=getIndex(oHist,obsCount,oLengthWeightset);
	actIndex=getIndex(aHist,actionCount,aLengthWeightset);
	if(obsPartSizeWeightset>0)
		hIndex=obsPartSizeWeightset*actIndex+obsIndex;
	else
		hIndex=actIndex;
	return hIndex;
}



#endif