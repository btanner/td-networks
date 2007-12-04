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

#include "TwoDimensionalMap.h"

#include <iostream>
#include <fstream>
#include <string>
using namespace::std;


int TwoDimensionalMap::countReachableStates() const {
	int r=0;
	for(int i=0;i<nrows_;i++)
		for(int j=0;j<ncols_;j++)
			if(get(i,j)!=1)
				r++;
			
	return r*4;
}

int TwoDimensionalMap::calcState(int row,int col, int dir){
	return dir*getCols()*getRows()+row*getRows()+col;
}

void TwoDimensionalMap::makeLookups(){
	
	int rStateCount=countReachableStates();
	lookups_=new int[getRows() *getCols()*4];
	for(int i=0;i<nrows_*ncols_*4;i++){
		lookups_[i]=-2;
	}
	
	rLookupR_=new int[rStateCount];
	rLookupC_=new int[rStateCount];
	rLookupD_=new int[rStateCount];
	

	
	int sID=0;
	for(int i=0;i<nrows_;i++)
		for(int j=0;j<ncols_;j++)
				for(int d=0;d<4;d++){
					if(get(i,j)!=1){
						lookups_[calcState(i,j,d)]=sID;
						rLookupR_[sID]=i;
						rLookupC_[sID]=j;
						rLookupD_[sID]=d;						
						sID++;
						if(sID>rStateCount){
							printf("Error; the new state count %d is more than the number of calculated reachable states %d\n",sID,rStateCount);
						}
					}else{
						lookups_[calcState(i,j,d)]=-1;
					}
				}
}


TwoDimensionalMap::TwoDimensionalMap(const TwoDimensionalMap& m){
	nrows_=m.getRows();
	ncols_=m.getCols();
	data_ = new int[m.getRows() * m.getCols()];
	label_ = new char[m.getRows() * m.getCols()*4];
	int rStateCount=m.countReachableStates();
	lookups_=new int[nrows_*ncols_*4];
	rLookupR_=new int[rStateCount];
	rLookupC_=new int[rStateCount];
	rLookupD_=new int[rStateCount];
	
	for(int i=0;i<rStateCount;i++){
		rLookupR_[i]=m.rLookupR_[i];
		rLookupC_[i]=m.rLookupC_[i];
		rLookupD_[i]=m.rLookupD_[i];
	}
	
	for(int i=0;i<nrows_*ncols_*4;i++){
		lookups_[i]=m.lookups_[i];
	}
	
	

	for(int i=0;i<nrows_;i++)
		for(int j=0;j<ncols_;j++){
			get(i,j)=m(i,j);
			for(int d=0;d<4;d++)
				getLabel(i,j,d)=m.getLabel(i,j,d);
			
		}
}

TwoDimensionalMap::TwoDimensionalMap(string fileName){
	int rows,cols;
	aCount_=4;
	if(SENSORFUSION)
		oCount_=8;
	else
		oCount_=2;
	
	ifstream fileIN(fileName.c_str());
			
	
	if(fileIN.good()){
		cout<<"Reading in file: "<<fileName<<endl;
		fileIN >> rows >>cols;
		cout<<"File Details:"<<endl<<"\t\t"<<"Rows: "<<rows<<" Cols: "<<cols<<" Observations: "<<oCount_<<" Actions: "<<aCount_<<endl;
			if (rows == 0 || cols == 0)
				throw BadSize();
		else{
			nrows_=rows;
			ncols_=cols;
		}
			data_ = new int[nrows_ * ncols_];
		label_= new char[nrows_*ncols_*4];
		
		for(unsigned int i=0;i<rows;i++){
			for (unsigned int j=0; j<cols; j++) {
					fileIN>>get(i,j);
			}
		}

		for(unsigned int i=0;i<rows;i++){
			for (unsigned int j=0; j<cols; j++) {
				for(unsigned int d=0;d<4;d++){
					fileIN>>getLabel(i,j,d);
//					cout<<getLabel(i,j,d)<<endl;
				}
			}
//			cout<<endl;
		}
		
		cout<<"Map looks like:"<<endl;
		for(unsigned int i=0;i<rows;i++){
			for (unsigned int j=0; j<cols; j++) {
				if(get(i,j)==1)
					cout<<"B";
				else
					if(get(i,j)==2)
						cout<<"C";
				else
					cout<<" ";
			}
			cout<<endl;
		}
		cout<<"Map looks like:"<<endl;
		for(unsigned int i=0;i<rows;i++){
			for (unsigned int j=0; j<cols; j++) {
				for(unsigned int d=0;d<4;d++)
					cout<<getLabel(i,j,d);
			}
			cout<<endl;
		}
		cout<<"Making lookups..."<<endl;
		makeLookups();
		cout<<"Lookups done"<<endl;
		
	}else{
		cout<<"Could not open file: "<<fileName<<". And you know why? It's not GOOD!"<<endl;	
		exit(1);
	}	
}

TwoDimensionalMap::TwoDimensionalMap(unsigned nrows, unsigned ncols)
: nrows_ (nrows)
, ncols_ (ncols)
//, data_ <--initialized below (after the 'if/throw' statement)
{
	if (nrows == 0 || ncols == 0)
		throw BadSize();
	data_ = new int[nrows * ncols];
	label_ = new char[nrows * ncols];
	cout<<"Bad constructor called; the one that takes rows and cols... there is much more to know than that..."<<endl;
}

TwoDimensionalMap::~TwoDimensionalMap()
{
	delete[] data_;
	delete[] lookups_;
	delete[] rLookupR_;
	delete[] rLookupC_;
	delete[] rLookupD_;
	delete[] label_;
} 
