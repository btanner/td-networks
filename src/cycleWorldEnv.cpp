/*
 *  cycleWorldEnv.cpp
 *  TDNetsCPP
 *
 *  Created by Brian Tanner on 07/09/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */

#include "cycleWorldEnv.h"
#include <iostream>
using std::cout;
using std::endl;

cycleWorldEnv::cycleWorldEnv(int Size) : 
pomdpEnv(Size,2,1){
	
//Exact same code as the ring world actually, with only one action.  What would it mean to have 3 actions?  crazy	
	belief[0]=1.0;
	for(int i=1;i<stateCount;i++){
		belief[i]=0.0;
	}
	for(int i=0;i<stateCount;i++){
		beliefTemp[i]=0.0;
	}
	
	//Initialize the transition matrix
	for(int a=0;a<actionCount;a++){
		for(int s=0;s<stateCount;s++){
			for(int s1=0;s1<stateCount;s1++){
				//					cout<<"Setting transitions to 0 for : "<<a<<" "<<s<<" "<<s1<<" Index: "<<calcIndex(a,s,s1)<<endl;
				transitions[calcIndex(a,s,s1)]=0.0;
			}
		}
	}
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
			observations[calcIndex(s,1)]=0.0f;
			observations[calcIndex(s,0)]=1.0f;
		}
		else{
			observations[calcIndex(s,1)]=1.0f;
			observations[calcIndex(s,0)]=0.0f;
		}
		
	}
	
	for(int s=0;s<stateCount;s++){
		oSample[s]=0.0;
	}
}
