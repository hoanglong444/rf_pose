cmake_minimum_required(VERSION 2.6)
PROJECT( head_pose )
FIND_PACKAGE( OpenCV REQUIRED )

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-g -pg -lpthread -std=gnu++0x -Wall")
    set(CMAKE_SHARED_LINKER_FLAGS "-pg")
endif()

ADD_LIBRARY(RandomForest
    src/CRTree.cpp
    src/CRForest.cpp
    src/FeatureExtractor.cpp
    src/DatasetLoader.cpp)
TARGET_LINK_LIBRARIES(RandomForest ${OpenCV_LIBS})

ADD_EXECUTABLE(run_training src/RunTraining.cpp)
TARGET_LINK_LIBRARIES(run_training ${OpenCV_LIBS} RandomForest)

ADD_EXECUTABLE(run_testing src/PoseEstimator.cpp src/RunTesting.cpp)
TARGET_LINK_LIBRARIES(run_testing ${OpenCV_LIBS} RandomForest)
