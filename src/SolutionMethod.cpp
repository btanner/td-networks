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

#include "SolutionMethod.h"
#include "TDNetAlgorithm.h"
#include <fstream>
#include <string>

using std::string;

SolutionMethod::SolutionMethod(SolutionType theType, float stepSizeVal) :
	theSolutionType(theType), stepSize(stepSizeVal) {
}

SolutionMethod::~SolutionMethod() {
}

void SolutionMethod::saveToFile(string fileName) const {
	ofstream outFile;
	outFile.open(fileName.c_str(), std::ios::binary);
	outFile<<theSolutionType<<" "<<stepSize;
	Save(outFile);

	outFile.close();
}

float SolutionMethod::getStepSize() const {
	return stepSize;
}

void SolutionMethod::setStepSize(float stepSizeVal) {
	stepSize=stepSizeVal;
}

SolutionMethod *SolutionMethod::loadFromFile(string fileName) {
	SolutionMethod *rSolution =NULL;
	ifstream inFile;
	float stepSize;
	inFile.open(fileName.c_str(), std::ios::binary);
	int theType;
	inFile>>theType>>stepSize;
	SolutionType theSolutionType=(SolutionType)theType;
	if (theSolutionType==TDNetAlgSolType) {
		cout<<"The file says its a TDNetAlgSol!"<<endl;
		rSolution=new TDNetAlgorithm(inFile, stepSize);
	} else {
		cout
				<<"Tried to load an algorithm from a stored representation that is of unknown type."
				<<endl;
	}

	inFile.close();
	return rSolution;
}

