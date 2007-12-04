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


#ifndef TWODIMENSIONALMAP_H
#define TWODIMENSIONALMAP_H
#include <string>
#include <iostream>

#include "staticInits.h"

using std::string;
using std::cout;
using std::endl;

class TwoDimensionalMap {
public:
	TwoDimensionalMap(string fileLocation);
	TwoDimensionalMap(unsigned nrows, unsigned ncols);
	int countReachableStates() const;
	int calcState(int row,int col, int dir);
	void makeLookups();
		
	
	
	// Throws a BadSize object if either size is zero
	class BadSize { };
	
	// Based on the Law Of The Big Three:
	~TwoDimensionalMap();
	TwoDimensionalMap(const TwoDimensionalMap& m);
	TwoDimensionalMap& operator= (const TwoDimensionalMap& m);
	
	// Access methods to get the (i,j) element:
	int&       operator() (unsigned i, unsigned j);        //subscript operators often come in pairs
	const int& operator() (unsigned i, unsigned j) const;  // subscript operators often come in pairs

	char&       getLabel(unsigned i, unsigned j,unsigned d);        //subscript operators often come in pairs
	const char& getLabel(unsigned i, unsigned j,unsigned d) const;  // subscript operators often come in pairs

	int&       get(unsigned i, unsigned j);        //subscript operators often come in pairs
	const int& get(unsigned i, unsigned j) const;  // subscript operators often come in pairs

	// These throw a BoundsViolation object if i or j is too big
			class BoundsViolation { };
			
			 int getRows() const;
			 int getCols() const;
			 int getRowForState(int State);
			 int getColForState(int State);
			 int getDirForState(int State);
			 int calcHashState(int row, int col, int dir) const;
			 int getLogicalState(int row, int col, int dir);
			 int getACount();
			 int getOCount();
			 char& getLabelForState(int logicalState);

private:
			unsigned nrows_, ncols_;
			char* label_;
		int* data_;
		int* lookups_;
		int* rLookupR_;
		int* rLookupC_;
		int* rLookupD_;
		int aCount_;
		int oCount_;
		
};

inline int TwoDimensionalMap::getACount(){
	return aCount_;	
}
inline int TwoDimensionalMap::getOCount(){
	return oCount_;	
}

inline int TwoDimensionalMap::getLogicalState(int row,int col, int dir){
	return(lookups_[calcHashState(row, col, dir)]);
}


inline int TwoDimensionalMap::calcHashState(int row,int col, int dir) const{
	return dir*getCols()*getRows()+row*getRows()+col;
}

inline char& TwoDimensionalMap::getLabelForState(int logicalState){
	int r=getRowForState(logicalState);
	int c=getColForState(logicalState);
	int d=getDirForState(logicalState);
	return getLabel(r,c,d);
	
}
inline int TwoDimensionalMap::getRowForState(int State) {
	return rLookupR_[State];
}
inline int TwoDimensionalMap::getColForState(int State) {
	return rLookupC_[State];
}
inline int TwoDimensionalMap::getDirForState(int State) {
	return rLookupD_[State];
}

inline int TwoDimensionalMap::getRows() const{
	return nrows_;
}

inline int TwoDimensionalMap::getCols() const{
	return ncols_;
}

inline int& TwoDimensionalMap::operator() (unsigned row, unsigned col)
{
	return get(row,col);
}

inline int& TwoDimensionalMap::get(unsigned row, unsigned col)
{
	if (row >= nrows_ || col >= ncols_){ 
		cout<<"Someone tried to get at row: "<<row<<" col: "<<col<<" size is : "<<nrows_<<" x "<<ncols_<<endl;throw BoundsViolation();
	}
	return data_[row*ncols_ + col];
}

inline char& TwoDimensionalMap::getLabel(unsigned row, unsigned col, unsigned dir)
{
	if (row >= nrows_ || col >= ncols_){ cout<<"Someone tried to get at row: "<<row<<" col: "<<col<<" size is : "<<nrows_<<" x "<<ncols_<<endl;throw BoundsViolation();}
	return label_[calcHashState(row,col,dir)];
}



inline const int& TwoDimensionalMap::operator() (unsigned row, unsigned col) const
{
	return get(row,col);
}
inline const int& TwoDimensionalMap::get(unsigned row, unsigned col) const
{
	if (row >= nrows_ || col >= ncols_){ cout<<"Someone tried to get at row: "<<row<<" col: "<<col<<" size is : "<<nrows_<<" x "<<ncols_<<endl;throw BoundsViolation();}
	return data_[row*ncols_ + col];
}
inline const char& TwoDimensionalMap::getLabel(unsigned row, unsigned col, unsigned dir) const
{
	if (row >= nrows_ || col >= ncols_){ cout<<"Someone tried to get at row: "<<row<<" col: "<<col<<" size is : "<<nrows_<<" x "<<ncols_<<endl;throw BoundsViolation();}
	return label_[calcHashState(row,col,dir)];
}

#endif