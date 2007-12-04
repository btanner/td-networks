/*
 *  ringWorldPOMDP.cpp
 *  TDNetsCPP
 *
 *  Created by Brian Tanner on 02/09/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */

#include "ringWorldEnv.h"
#include <iostream>
using std::cout;
using std::endl;

ringWorldEnv::ringWorldEnv(int Size) : 
	pomdpEnv(Size,2,2){
	

//set our initial state
		belief[0]=1.0;
			
		//Now create the 1 step transitions
		for(int s=stateCount-1;s>0;s--){
			transitions[calcIndex(0,s,s-1)]=1.0;
		}
		transitions[calcIndex(0,0,stateCount-1)]=1.0;
		for(int s=0;s<stateCount;s++){
			transitions[calcIndex(1,s,(s+1)%stateCount)]=1.0;
		}
		
		//Do observation setup
		for(int s=0;s<stateCount;s++){
			if(s==0){
					observations[calcIndex(s,1)]=1.0;
					observations[calcIndex(s,0)]=0.0;
			}
			else{
				observations[calcIndex(s,0)]=1.0;
				observations[calcIndex(s,1)]=0.0;
			}
			}
		
}
		
			