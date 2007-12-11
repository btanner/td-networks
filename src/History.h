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
	History(int actionCount, int obsCount, char startParam='a',
			int featureLength=1, int weightsetLength=1);
	History();
	virtual ~History();
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


	int featureIndex;
	int weightsetIndex;

	int actionCount;
	int obsCount;
	char startParam;
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

inline int History::getWeightsetIndex() const {
	return weightsetIndex;
}

inline int History::getFeatureIndex() const {
	//sanity check necessary?
	return featureIndex;
}

inline int History::getWeightsetSize() const {
	return totalWeightsetSize;
}
inline int History::getFeatureSize() const {
	return totalFeatureSize;
}

inline void History::setWeightsetIndex(int val) {
	weightsetIndex=val;
}

inline void History::setFeatureIndex(int val) {
	featureIndex=val;
}

inline int History::calcFeatureIndex() const {
	int obsIndex;
	int actIndex;
	int hIndex;

	obsIndex=getIndex(oHist, obsCount, oLengthFeature);
	actIndex=getIndex(aHist, actionCount, aLengthFeature);

	if (obsPartSizeFeature>0)
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

	obsIndex=getIndex(oHist, obsCount, oLengthWeightset);
	actIndex=getIndex(aHist, actionCount, aLengthWeightset);
	if (obsPartSizeWeightset>0)
		hIndex=obsPartSizeWeightset*actIndex+obsIndex;
	else
		hIndex=actIndex;
	return hIndex;
}

#endif
