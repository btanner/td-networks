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

#ifndef CROSSENTROPYONESTEPNODE_H
#define CROSSENTROPYONESTEPNODE_H
#include "OneStepNode.h"
#include <fstream>

using namespace std;
class CrossEntropyOneStepNode : public OneStepNode {
public:
	virtual float activate(float P) const;
	CrossEntropyOneStepNode(int obsToPredict, int actCond, Node *parent,
			int mcTargetTimeStep);
	CrossEntropyOneStepNode(ifstream &i);
	virtual ~CrossEntropyOneStepNode();
};

inline CrossEntropyOneStepNode::CrossEntropyOneStepNode(ifstream &i) :
	OneStepNode(i) {
}

#endif
