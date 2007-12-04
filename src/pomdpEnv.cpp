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

#include "pomdpEnv.h"
#include "convert.h"
#include <iostream>
#include <stdlib.h>
#include <cmath>

using std::cout;
using std::endl;
using std::string;

void pomdpEnv::checkConsistency(){
	printf("Checking POMDP Consistency....\n");
	bool good=true;
	
	for(int a=0;a<getActionCount();a++){
	for(int S1=0;S1<stateCount;S1++){
		float sum=0.0f;
		for(int S2=0;S2<stateCount;S2++){
			sum=sum+getTransition(a,S1,S2);
		}	
		if(fabs(sum-1.0f)>.000001){
			printf("Transitions or state %d action %d did not sum to 1, summed to %f\n",S1,a,sum);
			good=false;
		}
	}
		
	}
	if(!good){
		printf("Some part of the POMDP appears to be inconsistent (transitions)...quitting\n");
		exit(1);
		}
	else
		printf("POMDP appears consistent in transitions at least....checking observations\n");
		
	good=true;
	for(int S1=0;S1<stateCount;S1++){
		float obsSum=0.0f;
		for(int o=0;o<observationCount;o++){
			obsSum=obsSum+getObservation(S1,o);
		}	
		if(fabs(obsSum-1.0f)>.00001){
			printf("Observations for state %d  did not sum to 1, summed to %f\n",S1,obsSum);
			good=false;
		}
	}
	if(!good){
		printf("Some part of the POMDP appears to be inconsistent (observations)...quitting\n");
		exit(1);
		}
	else
		printf("POMDP is A-OK!\n");
	
	
	
}

float *pomdpEnv::getBeliefState(){
	return belief;	
}
pomdpEnv::pomdpEnv(int sValue, int oValue, int aValue):
Environment(oValue,aValue),stateCount(sValue)
{
	
	transitions = new float[aValue*sValue*sValue];
	observations = new float[sValue*sValue];
	printf("I just made %d belief state\n",sValue);
	printf("I just made %d in oSample\n",oValue);
	belief = new float[sValue];
	beliefTemp=new float[sValue];
	beliefTemp1=new float[sValue];
	oSample=new float[oValue];
	
	for(int i=0;i<sValue;i++){
		belief[i]=0.0;
		beliefTemp[i]=0.0;
		beliefTemp1[i]=0.0;
	}
	
	//Initialize the transition matrix
	for(int a=0;a<aValue;a++){
		for(int s=0;s<sValue;s++){
			for(int s1=0;s1<sValue;s1++){
				transitions[calcIndex(a,s,s1)]=0.0;
			}
		}
	}
	//Do observation setup
	for(int s=0;s<sValue;s++)
		for(int a=0;a<aValue;a++)
			observations[calcIndex(s,a)]=0;
	
	for(int s=0;s<oValue;s++){
		printf("Appparently %d is a bad place in oSample\n",s);
		oSample[s]=0.0;
	}
	
	
}

pomdpEnv::~pomdpEnv(){
	printf("For some reason the destructor is called\n");
	delete(transitions);
	delete(observations);
	delete(belief);
	delete(beliefTemp);
	delete(beliefTemp1);
	delete(oSample);
}




string pomdpEnv::toString() const{
	string returnString;
	returnString="Transition Matrix:\n-------------------------\n";

	//Initialize the transition matrix
//	for(int a=0;a<actionCount;a++){
//		returnString+="Action: "+stringify(a)+":\n";
//		for(int s=0;s<stateCount;s++){
//			for(int s1=0;s1<stateCount;s1++){
//				returnString+=stringify(transitions[calcIndex(a,s,s1)]);
//			}
//			returnString+="\n";
//		}
//		returnString+="\n";
//	}
		
		returnString="Observation Matrix:\n-------------------------\n";
		for(int s=0;s<stateCount;s++){
			returnString+="State: "+stringify(s);
			for(int o=0;o<observationCount;o++){
				returnString+=" O "+stringify(o)+" Pr=("+stringify(observations[calcIndex(s,o)])+") ";
			}
			returnString+="\n";
		}
	
//	returnString="Parent:\t"+stringify(parent);
//	returnString+="\tObsToPredict:\t "+stringify(obsToPredict);
//	returnString+="\tLevel:\t"+stringify(mcTargetTimeStep);
//	returnString+="\tActCond:\t"+stringify(actCond);
	return returnString;
	}

void pomdpEnv::updateBelief(int action, int observation){
	float tmpsum;
	for(int bs=0;bs<stateCount;bs++){
		tmpsum=0.0;
		for(int s=0;s<stateCount;s++){
			tmpsum+=transitions[calcIndex(action,s,bs)]*belief[s];
		}
		beliefTemp[bs]=tmpsum;
	}	
	
	for(int bs=0;bs<stateCount;bs++)
		belief[bs]=beliefTemp[bs];

	for(int bs=0;bs<stateCount;bs++)
		belief[bs]=observations[calcIndex(bs,observation)]*belief[bs];
	
	tmpsum=0;
	for(int bs=0;bs<stateCount;bs++)
		tmpsum+=belief[bs];
	for(int bs=0;bs<stateCount;bs++){
		belief[bs]/=tmpsum;
	}

}

int pomdpEnv::sampleObservation(int action){
	int o;
	float randsam,tmpsum;
	o=-1;
	
	//Before we sample the observation, we need to pretend move the belief state ahead based on the action
	for(int bs=0;bs<stateCount;bs++){
		tmpsum=0.0;
		for(int s=0;s<stateCount;s++){
			tmpsum+=transitions[calcIndex(action,s,bs)]*belief[s];
		}
		beliefTemp[bs]=tmpsum;
	}
	
	for(int i=0;i<observationCount;i++)
		oSample[i]=0.0;
	for(int i=0;i<observationCount;i++){
		for(int s=0;s<stateCount;s++){
			oSample[i]+=observations[calcIndex(s,i)]*beliefTemp[s];
		}
	}
	randsam=random()/(1.0*RAND_MAX);
	tmpsum=0.0;
	for(int i=0;i<observationCount;i++){
		tmpsum+=oSample[i];
		if(randsam<=tmpsum){
			o=i;
			break;
		}
	}
	//	printf("Sampled Observation %d for action %d\n",o,action);
	if(o<0){
//		printf("didnt get a valid observation... osample was %f %f and randsam was %f\n",__osample[0], __osample[1], randsam);
		cout<<"Error, didin't sample a valid observation\n"<<endl;
		exit(1);
	}
	return o;
}

float pomdpEnv::getProbOfObservationGivenAction(int targetObservation,int action){
	float tmpsum;
	//Before we sample the observation probability, we need to pretend move the belief state ahead based on the action
	for(int bs=0;bs<stateCount;bs++){
		tmpsum=0.0;
		for(int s=0;s<stateCount;s++){
			tmpsum+=transitions[calcIndex(action,s,bs)]*belief[s];
		}
		beliefTemp[bs]=tmpsum;
	}
	
	oSample[targetObservation]=0.0;
	for(int s=0;s<stateCount;s++){
		oSample[targetObservation]+=observations[calcIndex(s,targetObservation)]*beliefTemp[s];
	}
	return oSample[targetObservation];	
}

float pomdpEnv::getProbOfObservationGivenActions(int targetObservation,int a1, int a2){
	float tmpsum;
	//Before we sample the observation probability, we need to pretend move the belief state ahead based on the action
	for(int bs=0;bs<stateCount;bs++){
		beliefTemp[bs]=0.0f;
		beliefTemp1[bs]=0.0f;
	}

	//Calculate the new belief state if we took action a1
	for(int bs=0;bs<stateCount;bs++){
		tmpsum=0.0;
		for(int s=0;s<stateCount;s++){
			tmpsum+=transitions[calcIndex(a1,s,bs)]*belief[s];
		}
		beliefTemp[bs]=tmpsum;
	}
	for(int bs=0;bs<stateCount;bs++)
		beliefTemp1[bs]=beliefTemp[bs];
	//Calculate the new belief state if we took action a2
	for(int bs=0;bs<stateCount;bs++){
		tmpsum=0.0;
		for(int s=0;s<stateCount;s++){
			tmpsum+=transitions[calcIndex(a2,s,bs)]*beliefTemp1[s];
		}
		beliefTemp[bs]=tmpsum;
	}
	

	
	oSample[targetObservation]=0.0;
	for(int s=0;s<stateCount;s++){
		oSample[targetObservation]+=observations[calcIndex(s,targetObservation)]*beliefTemp[s];
	}
	return oSample[targetObservation];
	
}



