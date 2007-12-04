/*
 *  Serializable.h
 *  TDNetsCPP
 *
 *  Created by Brian Tanner on 07/11/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef SERIALIZABLE_H
#define SERIALIZABLE_H

#include <iostream>
#include <string>
#include <fstream>
using namespace std;

class Serializable {
public:
	virtual void Save(ofstream &o)const =0;
};

#endif