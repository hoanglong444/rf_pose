cmake_minimum_required(VERSION 2.6)
PROJECT( head_pose )
FIND_PACKAGE( OpenCV REQUIRED )

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-O2 -lpthread -std=gnu++0x -Wall")
endif()

ADD_EXECUTABLE( head_pose src/FeatureExtractor.h src/FeatureExtractor.cpp src/TestPatchExtract.cpp )
TARGET_LINK_LIBRARIES( head_pose ${OpenCV_LIBS} )




