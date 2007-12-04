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

#ifndef INPUT_H
#define INPUT_H

#include <vector>
#include "History.h"

using std::vector;

class Input{
public:
	static int timesInited;
	int timeStep;
	int historyNumber;
	int AOSelector;
	vector<float> *yt;
	Input(int timeStepValue, History const &H, vector<float> *predictions);
//Used now that we are basically making dummy inputs
	Input();
	Input *set(int timeStepValue, History const &H, vector<float> *predictions);
};
	
#endif