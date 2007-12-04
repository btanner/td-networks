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
	HistoryAlgSolType,
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