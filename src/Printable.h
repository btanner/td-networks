/*
 *  Printable.h
 *  TDNetsCPP
 *
 *  Created by Brian Tanner on 30/08/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef PRINTABLE_H
#define PRINTABLE_H

#include <iostream>
#include <string>

using std::string;

class Printable {
public:
	virtual string toString() const = 0;  // = 0 means it is "pure virtual"
};
std::ostream & operator<< (std::ostream & os, const Printable& someClass);

#endif