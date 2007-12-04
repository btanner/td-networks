/*
 *  TwoDimensionalEnv.cpp
 *  TDNetsCPP
 *
 *  Created by Brian Tanner on 07/11/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */

#include "TwoDimensionalEnv.h"
#include "staticInits.h"
#include <iostream>
using std::cout;
using std::endl;


string TwoDimensionalEnv::toString() const{
	string returnString;
	returnString="toString just a stub in TwoDimensionalEnv";
	return returnString;
}


void TwoDimensionalEnv::nextState(int Action,int Observation){
	
				int rightDir=(cDir+1)%4;
				int leftDir=cDir-1;
				if(leftDir<0)leftDir=3;
				
				if(Action==TURN_LEFT)
					cDir=leftDir;
				
				if(Action==TURN_RIGHT)
					cDir=rightDir;
				if(Action==GO_FORWARD){
					int S2Row=cRow;
					int S2Col=cCol;
					if(cDir==DIR_UP)
						S2Row=cRow-1;
					if(cDir==DIR_DOWN)
						S2Row=cRow+1;
					if(cDir==DIR_LEFT)
						S2Col=cCol-1;
					if(cDir==DIR_RIGHT)
						S2Col=cCol+1;
					
					if(S2Row>=theMap.getRows()||S2Row<0||S2Col<0||S2Col>=theMap.getCols()){
						S2Row=cRow;
					}else{
						if(theMap(S2Row,S2Col)!=0)
							S2Row=cRow;
					}
					if(S2Col>=theMap.getCols()||S2Row>=theMap.getRows()||S2Col<0){
						S2Col=cCol;
					}else{
						if(theMap(S2Row,S2Col)!=0)
							S2Col=cCol;
					}
					cRow=S2Row;
					cCol=S2Col;
				}
}

float TwoDimensionalEnv::getProbOfObservationGivenAction(int targetObs,int Action){
	if(getObservation(Action)==targetObs)
		return 1.0f;
	else
		return 0.0f;
}

int TwoDimensionalEnv::getObservationForState(int Row, int Col, int Dir){
	//We'll set it up by saying "what would we see if we went forward"
				int o;
				int S2Row=Row;
				int S2Col=Col;
				
				if(Dir==DIR_UP)
					S2Row=Row-1;
				if(Dir==DIR_DOWN)
					S2Row=Row+1;
				if(Dir==DIR_LEFT)
					S2Col=Col-1;
				if(Dir==DIR_RIGHT)
					S2Col=Col+1;
				
				//First just check if its safe to check MAP, because if we are outside of the boundaries its not
				if(S2Row<0||S2Row>=theMap.getRows()||S2Col<0||S2Col>=theMap.getCols()){
					o=1;
				}else{
					o=theMap(S2Row,S2Col);
				}
				
				return o;
}


int TwoDimensionalEnv::getObservationForCurrentState(){
	return getObservationForState(cRow,cCol,cDir);
}

int TwoDimensionalEnv::getObservation(int Action){
	//Figure out what the next state would be if we took the action
				int newRow=cRow;
				int newCol=cCol;
				int newDir=cDir;
				
				int rightDir=(cDir+1)%4;
				int leftDir=cDir-1;
				if(leftDir<0)leftDir=3;
				
				if(Action==TURN_LEFT)
					newDir=leftDir;
				
				if(Action==TURN_RIGHT)
					newDir=rightDir;
				
				if(Action==GO_FORWARD){
					int S2Row=cRow;
					int S2Col=cCol;
					if(cDir==DIR_UP)
						S2Row=cRow-1;
					if(cDir==DIR_DOWN)
						S2Row=cRow+1;
					if(cDir==DIR_LEFT)
						S2Col=cCol-1;
					if(cDir==DIR_RIGHT)
						S2Col=cCol+1;
					
					if(S2Row>=theMap.getRows()||S2Row<0||S2Col<0||S2Col>=theMap.getCols()){
						S2Row=cRow;
					}else{
						if(theMap(S2Row,S2Col)!=0)
							S2Row=cRow;
					}
					if(S2Col>=theMap.getCols()||S2Row>=theMap.getRows()||S2Col<0){
						S2Col=cCol;
					}else{
						if(theMap(S2Row,S2Col)!=0)
							S2Col=cCol;
					}
					newRow=S2Row;
					newCol=S2Col;
				}
				return getObservationForState(newRow,newCol,newDir);
}

TwoDimensionalEnv::TwoDimensionalEnv(TwoDimensionalMap &Map):
Environment(2,2),theMap(Map){
	
	cout<<"First, setting initial state"<<endl;
	bool setInitState=false;
	for(int i=0;i<theMap.getRows()&&!setInitState;i++){
		for(int j=0;j<theMap.getCols()&&!setInitState;j++){
			//If we havent' set the initial state yet AND this grid square is not blocked, set it
			if(!setInitState&&theMap(i,j)==0){
				cRow=i;
				cCol=j;
				cDir=0;
				setInitState=true;
			}
		}
	}
}
