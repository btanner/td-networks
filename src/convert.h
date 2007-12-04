/*
 *  convert.h
 *  TDNetsCPP
 *
 *  Created by Brian Tanner on 30/08/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef CONVERT_H
#define CONVERT_H

// File: convert.h
#include <iostream>
#include <sstream>
#include <string>
#include <stdexcept>

class BadConversion : public std::runtime_error {
public:
	BadConversion(const std::string& s)
	: std::runtime_error(s)
{ }
};

inline std::string stringify(int x)
{
	std::ostringstream o;
	if (!(o << x))
		throw BadConversion("stringify(int)");
	return o.str();
}
inline std::string stringify(double x)
{
	std::ostringstream o;
	if (!(o << x))
		throw BadConversion("stringify(double)");
	return o.str();
}
inline std::string stringify(float x)
{
	std::ostringstream o;
	if (!(o << x))
		throw BadConversion("stringify(float)");
	return o.str();
}

#endif