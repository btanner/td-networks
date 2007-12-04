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

#include <iostream>
#include <vector>
#include <stdio.h>
#include <sstream> 
#include <string.h>
#include <math.h>
#include "TwoDimensionalPOMDP.h"
#include "TwoDimensionalEnv.h"
#include "TwoDimensionalMap.h"
#include "staticInits.h"


using std::cout;
using std::endl;
using std::vector;
using std::string;

int main (int argc, char * const argv[]) {
	cout<<"Starting program"<<endl;
	
	int envCount=2;
	vector<string> files(envCount);
	vector<string> labels(envCount);
	
	
/*	files[0]=string("POMDP_2_Matlab_files/pomdpMaps/IJCAITrain1.txt");
	files[1]=string("POMDP_2_Matlab_files/pomdpMaps/IJCAITrain2.txt");
	files[2]=string("POMDP_2_Matlab_files/pomdpMaps/IJCAITrain3.txt");
	files[3]=string("POMDP_2_Matlab_files/pomdpMaps/IJCAITest.txt");
	files[4]=string("POMDP_2_Matlab_files/pomdpMaps/IJCAITrain4.txt");
	
	labels[0]=string("Train1");
	labels[1]=string("Train2");
	labels[2]=string("Train3");
	labels[3]=string("Test");
	labels[4]=string("Train4");
*/
/*	files[0]=string("POMDP_2_Matlab_files/pomdpMaps/Village1.txt");
	labels[0]=string("Train1");

	files[1]=string("POMDP_2_Matlab_files/pomdpMaps/Village2.txt");
	labels[1]=string("Test");
*/
	files[0]=string("POMDP_2_Matlab_files/pomdpMaps/DoomTrain.txt");
	labels[0]=string("Train1");
	
	files[1]=string("POMDP_2_Matlab_files/pomdpMaps/DoomTest.txt");
	labels[1]=string("Test");

	
	for(int i=0;i<envCount;i++){

		TwoDimensionalMap Map(files[i]);
		TwoDimensionalPOMDP *PEnv= new TwoDimensionalPOMDP(Map);
		PEnv->checkConsistency();
		
		PEnv->writeToMatlabFile(labels[i]);
		
		cout<<"On env: "<<i<<" there are: "<<Map.countReachableStates()<<" states";
		delete(PEnv);
		
	}
	
	
	
}
	
	
	
	
