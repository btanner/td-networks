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