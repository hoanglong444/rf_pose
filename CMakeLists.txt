cmake_minimum_required(VERSION 2.6)

PROJECT( head_pose )
FIND_PACKAGE( OpenCV REQUIRED )
ADD_EXECUTABLE( head_pose src/CRPatch.h src/CRPatch.cpp src/CRTree.h src/CRTree.cpp src/CRForest.h )
TARGET_LINK_LIBRARIES( head_pose ${OpenCV_LIBS} )




