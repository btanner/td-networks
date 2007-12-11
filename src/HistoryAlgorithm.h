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
	virtual ~HistoryAlgorithm();
	
	virtual void Save(ofstream &o) const;
	
	virtual void updateState(int action, int observation);
	virtual float getPForAO(int action, int observation) const;

protected:
	/*Parameters we'll need to save.*/
	int historyLength;
	int t;
	int actionCount;
	int observationCount;
	float *numH;
	float *numOH;
	
	private:
	/*Parameters we'll need to recreate*/
//	float alpha;

	//Create the history structure
	History H;
};
#endif
