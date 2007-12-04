/*
 *  loadedPOMDPWorld.h
 *  TDNetsCPP
 *
 *  Created by Brian Tanner on 04/11/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef TwoDimensionalPOMDP_H
#define TwoDimensionalPOMDP_H


#include <string>
#include "TwoDimensionalMap.h"
using std::string;




#include "pomdpEnv.h"


#define DIR_UP 0
#define DIR_LEFT 3
#define DIR_RIGHT 1
#define DIR_DOWN 2


#define TILE 4
#define STICKYTILE 0
#define WALL 1
#define CARPET 2
#define PUDDLE 3

#define TILE_SLIP 0
#define CARPET_SLIP .1
#define PUDDLE_SLIP .2


//#define XSTOCH 1
#define XDET 1
//#define XDETOBS 1

#ifdef XSTOCH
#define STICKYTILE_STAY .03
#define STICKYTILE_NEXT .95
#define STICKYTILE_TWICE .02

#define O_WW .925f
#define O_WO .075f
#define O_OO .9f
#define O_OW .1f
#endif

#ifdef XDET
#define STICKYTILE_STAY 0.0f
#define STICKYTILE_NEXT 1.0f
#define STICKYTILE_TWICE 0.0f

#define O_WW 1.0f
#define O_WO 0.0f
#define O_OO 1.0f
#define O_OW 0.0f
#endif

#ifdef XDETOBS
#define STICKYTILE_STAY .03
#define STICKYTILE_NEXT .95
#define STICKYTILE_TWICE .02
#define O_WW 1.0f
#define O_WO 0.0f
#define O_OO 1.0f
#define O_OW 0.0f
#endif



class TwoDimensionalPOMDP: public pomdpEnv{
public:
	TwoDimensionalPOMDP(TwoDimensionalMap &MAP);
	void printCurrentBelief();
	void writeToMatlabFile(string mapName);
	
	
	
	
	
private:
	TwoDimensionalMap theMap;
	int calculateLogicalForwardState(int LogicalS1);
	int calculateLogicalLeftState(int LogicalS1);
	int calculateLogicalRightState(int LogicalS1);
	int calculateLogicalBackwardState(int LogicalS1);
		int getTileType(int LogicalS);
};


#endif