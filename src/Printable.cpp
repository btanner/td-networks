/*
 *  Printable.cpp
 *  TDNetsCPP
 *
 *  Created by Brian Tanner on 30/08/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */

#include "Printable.h"


std::ostream & operator<< (std::ostream & os,const Printable& someClass)
{
    return os << someClass.toString();
}