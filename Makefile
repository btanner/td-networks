#---------------------------------------------------------------------
# Project configuration.
#---------------------------------------------------------------------

# Name of build target.
TARGET := foobar

# Source directories (delimited with spaces).
BUILD_VPATH := src

# Source files to build target from. 
BUILD_FILES := CrossEntropyOneStepNode.cpp NodeFactory.cpp TDNet.cpp   TwoDimensionalPOMDP.cpp  pomdpEnv.cpp Environment.cpp   OneStepNode.cpp TDNetAlgorithm.cpp  Weights.cpp   predictionPointerFactory.cpp History.cpp    TDNetsMain.cpp   cycleWorldEnv.cpp  ringWorldEnv.cpp HistoryAlgorithm.cpp  Printable.cpp   TraceNode.cpp   forgetfulStateMemory.cpp Input.cpp   SigmoidOneStepNode.cpp  TwoDimensionalEnv.cpp  forgetfulTraceMemory.cpp Node.cpp   SolutionMethod.cpp  TwoDimensionalMap.cpp  inputPointerFactory.cpp

#There is another main program in POMDP_2_Matlab.cpp that can be used to convert a
#POMDP specified in Matlab to an environment that can be used for this code

# Build configuration [release] or [debug] or [profile].
BUILD_CONFIG := release

#---------------------------------------------------------------------
# Configure build enviroment.
#---------------------------------------------------------------------

include makefile.gcc

#---------------------------------------------------------------------
# The End.
#---------------------------------------------------------------------
