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
#ifndef SIGMOIDONESTEPNODE_H
#define SIGMOIDONESTEPNODE_H
#include "CrossEntropyOneStepNode.h"

#include <fstream>
using namespace::std;

class SigmoidOneStepNode : public CrossEntropyOneStepNode {
public:
	virtual float inverseActivate(const float P) const;
	SigmoidOneStepNode(int obsToPredict, int actCond, Node *parent,
			int mcTargetTimeStep);
	SigmoidOneStepNode(ifstream &i);
	virtual ~SigmoidOneStepNode();
};

inline SigmoidOneStepNode::SigmoidOneStepNode(ifstream &i) :
	CrossEntropyOneStepNode(i) {
}
#endif
