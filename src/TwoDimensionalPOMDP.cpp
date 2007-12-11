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

#include "TwoDimensionalPOMDP.h"
#include "TwoDimensionalMap.h"
#include <string>
#include "staticInits.h"
#include <fstream>
#include <iostream>

using namespace std;

void TwoDimensionalPOMDP::writeToMatlabFile(string mapName) {
	ofstream outFile;
	string fileName="POMDP_2_Matlab_files/matlabEnvs/getMAPPomdp"+mapName+".m";
	outFile.open(fileName.c_str());

	outFile<<"function [res] = getMAPPomdp"<<mapName<<"()"<<endl;
	outFile
			<<"% This file is auto created by the RLAI Temporal Difference Networks framework"
			<<endl;
	outFile<<"res.S = "<<this->getStateCount()<<endl;
	outFile<<"res.A = "<<this->getActionCount()<<endl;
	outFile<<"res.O = "<<this->getObservationCount()<<endl;
	//Made this sparse so the file isn't huge and is readable
	outFile<<"res.prior = sparse(1,"<<this->getStateCount()<<");"<<endl;
	for (int i=0; i<this->getStateCount(); i++) {
		if (this->belief[i]>0.0f)
			outFile<<"res.prior(1,"<<(i+1)<<")="<<this->belief[i]<<";"<<endl;
	}

	outFile<<"res.transmat = cell(1,res.A);"<<endl;

	//Do Action i
	//Also made this sparse to make the files manageable and readable
	for (int a=0; a<this->getActionCount(); a++) {
		outFile<<"res.transmat{1,"<<(a+1)<<"} = sparse("<<this->getStateCount()
				<<","<<this->getStateCount()<<");"<<endl;
		for (int s=0; s<this->getStateCount(); s++) {
			for (int s1=0; s1<this->getStateCount(); s1++) {
				float theProb=transitions[calcIndex(a,s,s1)];
				if (theProb>0.0f) {
					outFile<<"res.transmat{1,"<<(a+1)<<"}("<<(s+1)<<","<<(s1+1)
							<<")="<<theProb<<";";
				}
			}
			outFile<<endl;
		}
	}

	//Also made this sparse to make the files manageable and readable
	outFile<<"res.obsmat = sparse("<<this->getStateCount()<<","
			<<this->getObservationCount()<<");"<<endl;

	for (int s=0; s<this->getStateCount(); s++) {
		for (int o=0; o<this->getObservationCount(); o++) {
			float theProb=observations[calcIndex(s,o)];
			if (theProb>0.0f)
				outFile<<"res.obsmat("<<(s+1)<<","<<(o+1)<<")="<<theProb<<";";
		}
		outFile << endl;
	}

	outFile<<"\%Mappings between nominal states and Row, Column, Direction"
			<<endl;
	outFile<<"res.map = [";
	for (int s=0; s<this->getStateCount(); s++) {
		outFile<<theMap.getRowForState(s)<<", "<<theMap.getColForState(s)<<", "
				<<theMap.getDirForState(s)<<";"<<endl;
	}
	outFile<<"];"<<endl;

	outFile<<"\%Mappings between nominal states and Labels"<<endl;
	outFile<<"res.Class = [";
	for (int s=0; s<this->getStateCount(); s++) {
		outFile<<"'"<<theMap.getLabelForState(s)<<"';";
	}
	outFile<<"];"<<endl;

	outFile<<"res.validStates=ones(res.S,1);"<<endl;
	outFile<<"res.validStates=res.validStates';"<<endl;
	outFile<<"for a = 1:res.A"<<endl;
	outFile<<"\t res.transmat{a}=sparse(res.transmat{a});"<<endl;
	outFile<<"for o = 1:res.O"<<endl;
	outFile<<"\t res.obs{a,o}=sparse(diag(eye(res.S)*res.obsmat(:,o)));"<<endl;
	outFile<<"end"<<endl;
	outFile<<"end"<<endl;
	outFile.close();
}

void TwoDimensionalPOMDP::printCurrentBelief() {
	int curState=0;
	float prob=0;
	float probSum=0;
	for (int i=0; i<stateCount; i++) {
		if (belief[i]>0) {
			curState=i;
			prob=belief[i];
			probSum=probSum+prob;
		}
	}
	if (probSum>1.0)
		cout<<"Added up probSums to: "<<probSum
				<<" means the belief state is corrupt"<<endl;

	int Row=theMap.getRowForState(curState);
	int Col=theMap.getColForState(curState);
	int Dir=theMap.getDirForState(curState);

	for (int i=0; i<theMap.getRows(); i++) {
		for (int j=0; j<theMap.getCols(); j++) {
			//Print the agent
			if (Row==i&&Col==j) {
				if (Dir==DIR_UP)
					cout<<"^ ";
				if (Dir==DIR_DOWN)
					cout<<"v ";
				if (Dir==DIR_LEFT)
					cout<<"< ";
				if (Dir==DIR_RIGHT)
					cout<<"> ";
			} else
				cout<<theMap(i, j)<<" ";

		}
		cout<<endl;
	}
}

int TwoDimensionalPOMDP::calculateLogicalLeftState(int LogicalS1) {
	int S1r=theMap.getRowForState(LogicalS1);
	int S1c=theMap.getColForState(LogicalS1);
	int S1d=theMap.getDirForState(LogicalS1);

	int leftDir=S1d-1;
	if (leftDir<0)
		leftDir=3;

	return theMap.getLogicalState(S1r, S1c, leftDir);
}

int TwoDimensionalPOMDP::calculateLogicalRightState(int LogicalS1) {
	int S1r=theMap.getRowForState(LogicalS1);
	int S1c=theMap.getColForState(LogicalS1);
	int S1d=theMap.getDirForState(LogicalS1);

	int rightDir=(S1d+1)%4;

	return theMap.getLogicalState(S1r, S1c, rightDir);
}

int TwoDimensionalPOMDP::calculateLogicalBackwardState(int LogicalS1) {
	//Turn twice then call calculateLogicalForwardState
	int TT=calculateLogicalRightState(calculateLogicalRightState(LogicalS1));
	int TTF=calculateLogicalForwardState(TT);
	int TTFTT=calculateLogicalRightState(calculateLogicalRightState(TTF));
	return TTFTT;

}

int TwoDimensionalPOMDP::calculateLogicalForwardState(int LogicalS1) {
	int S1i=theMap.getRowForState(LogicalS1);
	int S1j=theMap.getColForState(LogicalS1);
	int S1k=theMap.getDirForState(LogicalS1);

	int S2i=S1i;
	int S2j=S1j;
	int S2k=S1k;

	if (S1k==DIR_UP)
		S2i=S1i-1;
	if (S1k==DIR_DOWN)
		S2i=S1i+1;
	if (S1k==DIR_LEFT)
		S2j=S1j-1;
	if (S1k==DIR_RIGHT)
		S2j=S1j+1;

	if (S2i>=theMap.getRows()||S2i<0||S2j<0||S2j>=theMap.getCols()) {
		S2i=S1i;
	} else {
		if (theMap(S2i, S2j)==WALL)
			S2i=S1i;
	}
	if (S2j>=theMap.getCols()||S2i>=theMap.getRows()||S2j<0) {
		S2j=S1j;
	} else {
		if (theMap(S2i, S2j)==WALL)
			S2j=S1j;
	}
	return (theMap.getLogicalState(S2i, S2j, S2k));
}

int TwoDimensionalPOMDP::getTileType(int LogicalS) {
	int Si=theMap.getRowForState(LogicalS);
	int Sj=theMap.getColForState(LogicalS);

	return theMap(Si, Sj);
}

/*New Code */
TwoDimensionalPOMDP::TwoDimensionalPOMDP(TwoDimensionalMap &Map) :
	pomdpEnv(Map.countReachableStates(), Map.getOCount(), Map.getACount()),
			theMap(Map) {

	cout<<"Just sent to pomdpEnv constructor: "<<Map.countReachableStates()
			<<" "<<Map.getACount()<<" "<<Map.getOCount()<<endl;
	cout<<"The POMDP has: "<<this->getStateCount()<<" states"<<endl;

	cout<<"First, setting initial belief state"<<endl;

	//This used to be complicated, but is now easy because we know all of the valid states

	belief[0]=1.0;

	for (int S1=0; S1<this->getStateCount(); S1++) {
		int SRight=calculateLogicalRightState(S1);
		int SLeft=calculateLogicalLeftState(S1);
		int SForward=calculateLogicalForwardState(S1);
		int SBack=calculateLogicalBackwardState(S1);

		int SRightRight=calculateLogicalRightState(SRight);
		int SLeftLeft=calculateLogicalLeftState(SLeft);
		int SForwardForward=calculateLogicalForwardState(SForward);
		int SBackBack=calculateLogicalBackwardState(SBack);

		float pStay=0;
		float pNext=0;
		float pTwice=0;

		int tileType=getTileType(S1);

		if (tileType==CARPET) {
			pNext=1.0-CARPET_SLIP;
			pStay=CARPET_SLIP;
			pTwice=0;
		}
		if (tileType==PUDDLE) {
			pNext=1.0-PUDDLE_SLIP;
			pStay=PUDDLE_SLIP;
			pTwice=0;
		}
		if (tileType==TILE) {
			pNext=1.0-TILE_SLIP;
			pStay=TILE_SLIP;
			pTwice=0;
		}
		if (tileType==STICKYTILE) {
			pStay=STICKYTILE_STAY;
			pNext=STICKYTILE_NEXT;
			pTwice=STICKYTILE_TWICE;
		}

		//Write a bit of code to change the label to lower case IF we are facing away from a wall
		//We want to set the special character for being in a class AND being facing a wall here
		int T=calculateLogicalRightState(S1);
		int TT=calculateLogicalRightState(T);
		int TTF=calculateLogicalForwardState(TT);

		bool backToWall=false;
		if (TT==TTF)
			backToWall=true;

		bool isCorner=false;

		//Check if he would be in a corner for any of the 4 directions
		int StartPos=S1;
		for (int x=0; x<4&&!isCorner; x++) {
			int T=calculateLogicalRightState(StartPos);
			int TF=calculateLogicalForwardState(T);
			int TT=calculateLogicalRightState(T);
			int TTF=calculateLogicalForwardState(TT);
			if (TF==T&&TTF==TT)
				isCorner=true;
			StartPos=calculateLogicalRightState(StartPos);
		}
		//This is the code for the doom maps
		if (WALLSANDCORNERS) {

			char &theLabel=theMap.getLabelForState(S1);
			if (theLabel=='H') {
				if (backToWall)
					theLabel='h';
				if (isCorner)
					theLabel='a';
				if (isCorner&&backToWall)
					theLabel='b';
			}
			if (theLabel=='R') {
				if (backToWall)
					theLabel='r';
				if (isCorner)
					theLabel='c';
				if (isCorner&&backToWall)
					theLabel='d';
			}
			if (theLabel=='O') {
				if (backToWall)
					theLabel='o';
				if (isCorner)
					theLabel='e';
				if (isCorner&&backToWall)
					theLabel='f';
			}
		}

		//cout<<"States: "<<S1<<" "<<SLeft<<" "<<SLeftLeft<<" "<<SRight<<" "<<SRightRight<<" "<<SForward<<" "<<SForwardForward<<endl;		
		// Coming back to this code I don't know what this means.  When could turning left ever be equal to staying put?

		if (S1==SLeft)
			setTransition(TURN_LEFT,S1,SLeft,1.0);
		else {
			setTransition(TURN_LEFT,S1,SLeftLeft,pTwice);
			setTransition(TURN_LEFT,S1,SLeft,pNext);
			setTransition(TURN_LEFT,S1,S1,pStay);
		}

		if (S1==SRight)
			setTransition(TURN_RIGHT,S1,SRight,1.0);
		else {
			setTransition(TURN_RIGHT,S1,SRightRight,pTwice);
			setTransition(TURN_RIGHT,S1,SRight,pNext);
			setTransition(TURN_RIGHT,S1,S1,pStay);
		}

		if (S1==SForward) {
			//In this case all 3 forwards are the same state
			setTransition(GO_FORWARD,S1,SForward,1.0);
		} else {
			if (SForward==SForwardForward) {
				//In this case the next and 2nd states are same so they share the probability
				setTransition(GO_FORWARD,S1,S1,pStay);
				setTransition(GO_FORWARD,S1,SForward,pNext+pTwice);
			} else {
				setTransition(GO_FORWARD,S1,SForwardForward,pTwice);
				setTransition(GO_FORWARD,S1,SForward,pNext);
				setTransition(GO_FORWARD,S1,S1,pStay);
			}
		}

		if (S1==SBack) {
			//In this case all 3 backs are the same state
			setTransition(GO_BACKWARD,S1,SBack,1.0);
		} else {
			if (SBack==SBackBack) {
				//In this case the next and 2nd states are same so they share the probability
				setTransition(GO_BACKWARD,S1,S1,pStay);
				setTransition(GO_BACKWARD,S1,SBack,pNext+pTwice);
			} else {
				setTransition(GO_BACKWARD,S1,SBackBack,pTwice);
				setTransition(GO_BACKWARD,S1,SBack,pNext);
				setTransition(GO_BACKWARD,S1,S1,pStay);
			}
		}

		//Calculate observation for S1 ... if moving forward has no effect we are at a wall, otherwise it's open
		//Need to find out if there is a wall left, forward, and right of us

		int SRightForward=calculateLogicalForwardState(SRight);
		int SLeftForward=calculateLogicalForwardState(SLeft);

		//SENSORFUSION MIGHT BE BROKEN
		if (SENSORFUSION) {
			int FF = 0, LL = 1, RR = 2;
			float ObsProbs[3][2];
			if (SRight==SRightForward) {
				ObsProbs[RR][0]=O_WO;
				ObsProbs[RR][1]=O_WW;
			} else {
				ObsProbs[RR][0]=O_OO;
				ObsProbs[RR][1]=O_OW;
			}
			if (SLeft==SLeftForward) {
				ObsProbs[LL][0]=O_WO;
				ObsProbs[LL][1]=O_WW;
			} else {
				ObsProbs[LL][0]=O_OO;
				ObsProbs[LL][1]=O_OW;
			}
			if (SForward==SForwardForward) {
				ObsProbs[FF][0]=O_WO;
				ObsProbs[FF][1]=O_WW;
			} else {
				ObsProbs[FF][0]=O_OO;
				ObsProbs[FF][1]=O_OW;
			}

			for (int i=0; i<2; i++)
				for (int j=0; j<2; j++)
					for (int k=0; k<2; k++) {
						int ObsIndex=i+2*j+4*k;
						//ObsIndex goes from 0 to 7
						float combinedProb=ObsProbs[RR][i]*ObsProbs[LL][j]
								*ObsProbs[FF][k];
						setObservation(S1, ObsIndex, combinedProb);
						//				printf("Just set Prob of seeing %d %d %d to %f which is (%f * %f *%f)\n",i,j,k, combinedProb,ObsProbs[RR][i],ObsProbs[LL][j],ObsProbs[FF][k]);
					}
		}//End the sensor fusion if
		else {
			//Regular old 1/0 sensor
			float ObsProbs[3][2];

			//REalizedd this was broken 3 days before IJCAI, its been broken since the early days.
			//		if(SForward==SForwardForward){
			if (SForward==S1) {
				ObsProbs[0][0]=O_WO;
				ObsProbs[0][1]=O_WW;
			} else {
				ObsProbs[0][0]=O_OO;
				ObsProbs[0][1]=O_OW;
			}
			setObservation(S1, 0, ObsProbs[0][0]);
			setObservation(S1, 1, ObsProbs[0][1]);

		}
	}

	cout<<"\tDone setting up POMDP!"<<endl;
	//End of method

	/* I as printing out the map here with labels for viewign to see if they are right */
	cout<<"Map looks like:"<<endl;

	for (int i=0; i<theMap.getRows(); i++) {
		for (int j=0; j<theMap.getCols(); j++) {
			for (unsigned int d=0; d<4; d++)
				cout<<theMap.getLabel(i, j, d);
			cout<<" ";

		}
		cout<<endl;
	}

}

//Old code
/*
 TwoDimensionalPOMDP::TwoDimensionalPOMDP(TwoDimensionalMap &Map):
 pomdpEnv(Map.getCols()*Map.getRows()*4,2,2),theMap(Map){
 cout<<"Just sent to pomdpEnv constructor: "<<Map.getCols()*Map.getRows()*4<<" "<<2<<" "<<2<<endl;
 cout<<"Ok, so we have a 2D POMDP that was inited from a map?"<<endl;
 cout<<"The POMDP has: "<<this->getStateCount()<<" states"<<endl;
 
 cout<<"First, setting initial belief state"<<endl;
 bool setInitState=false;
 for(int i=0;i<theMap.getRows()&&!setInitState;i++){
 for(int j=0;j<theMap.getCols()&&!setInitState;j++){
 //If we havent' set the initial state yet AND this grid square is not blocked, set it
 if(!setInitState&&(theMap(i,j)==TILE||theMap(i,j)==CARPET)){
 printf("Apparently %d is not a valid state\n",calcState(i,j,0));
 belief[calcState(i,j,0)]=1;
 setInitState=true;
 }
 }
 }

 cout<<"Setting transition matrix for turns"<<endl;
 //Transition Matrix for TURN actions 
 for(int S1i=0;S1i<theMap.getRows();S1i++){
 for(int S1j=0;S1j<theMap.getCols();S1j++){
 for(int S1k=0;S1k<4;S1k++){

 int State1=calcState(S1i,S1j,S1k);
 int rightDir=(S1k+1)%4;

 int leftDir=S1k-1;
 if(leftDir<0)leftDir=3;

 int rightState=calcState(S1i,S1j,rightDir);
 int leftState=calcState(S1i,S1j,leftDir);
 //					printf("Bad index: %d which comes from %d %d %d\n",calcIndex(TURN_RIGHT,State1,rightState),TURN_RIGHT,State1,rightState);
 //Don't calculate this.
 //	transitions[calcIndex(TURN_RIGHT,State1,rightState)]=1.0;
 
 if(Map(S1i,S1j)==CARPET){
 transitions[calcIndex(TURN_LEFT,State1,leftState)]=1.0-CARPET_SLIP;
 transitions[calcIndex(TURN_LEFT,State1,State1)]=CARPET_SLIP;
 }else
 transitions[calcIndex(TURN_LEFT,State1,leftState)]=1.0;
 
 
 }
 }
 }

 cout<<"Setting transition matrix for forward"<<endl;
 //Transition Matrix for Forward action 
 for(int S1i=0;S1i<theMap.getRows();S1i++){
 for(int S1j=0;S1j<theMap.getCols();S1j++){
 for(int S1k=0;S1k<4;S1k++){
 int State1=calcState(S1i,S1j,S1k);
 
 int S2i=S1i;
 int S2j=S1j;
 if(S1k==DIR_UP)
 S2i=S1i-1;
 if(S1k==DIR_DOWN)
 S2i=S1i+1;
 if(S1k==DIR_LEFT)
 S2j=S1j-1;
 if(S1k==DIR_RIGHT)
 S2j=S1j+1;
 
 if(S2i>=Map.getRows()||S2i<0||S2j<0||S2j>=Map.getCols()){
 S2i=S1i;
 }else{
 if(Map(S2i,S2j)==WALL)
 S2i=S1i;
 }
 if(S2j>=Map.getCols()||S2i>=Map.getRows()||S2j<0){
 S2j=S1j;
 }else{
 if(Map(S2i,S2j)==WALL)
 S2j=S1j;
 }
 
 int State2=calcState(S2i,S2j,S1k);
 
 if(State1==State2)
 transitions[calcIndex(GO_FORWARD,State1,State2)]=1.0;
 else{
 if(Map(S1i,S1j)==CARPET){
 transitions[calcIndex(GO_FORWARD,State1,State2)]=1.0-CARPET_SLIP;
 transitions[calcIndex(GO_FORWARD,State1,State1)]=CARPET_SLIP;
 }else
 transitions[calcIndex(GO_FORWARD,State1,State2)]=1.0;
 }
 
 }
 }
 }
 
 //Observation Matrix
 //We'll set it up by saying "what would we see if we went forward"
 cout<<"Setting observation matrix"<<endl;
 for(int S1i=0;S1i<theMap.getRows();S1i++){
 for(int S1j=0;S1j<theMap.getCols();S1j++){
 for(int S1k=0;S1k<4;S1k++){
 int State1=calcState(S1i,S1j,S1k);
 
 int S2i=S1i;
 int S2j=S1j;
 if(S1k==DIR_UP)
 S2i=S1i-1;
 if(S1k==DIR_DOWN)
 S2i=S1i+1;
 if(S1k==DIR_LEFT)
 S2j=S1j-1;
 if(S1k==DIR_RIGHT)
 S2j=S1j+1;
 
 //First just check if its safe to check MAP, because if we are outside of the boundaries its not
 if(S2i<0||S2i>=Map.getRows()||S2j<0||S2j>=Map.getCols()){
 observations[calcIndex(State1,0)]=0.0f;
 observations[calcIndex(State1,1)]=1.0f;					
 }else{
 if(Map(S2i,S2j)==TILE||Map(S2i,S2j)==CARPET){
 observations[calcIndex(State1,0)]=1.0f;
 observations[calcIndex(State1,1)]=0.0f;
 }
 if(Map(S2i,S2j)==WALL){
 observations[calcIndex(State1,0)]=0.0f;
 observations[calcIndex(State1,1)]=1.0f;
 }				
 }
 }
 }
 }
 cout<<"\tDone setting up POMDP!"<<endl;
 //End of method
 }
 */
