#include "PoseEstimator.h"
#include <iostream>

int main(int argc, char* argv[])
{
    if (argc <= 1) {
        std::cerr << argv[0] << " filename" << std::endl;
        return -1;
    }

    CRForest forest("trees");
    PoseEstimator estimator(forest);

    estimator.estimate(argv[1]);

    return 0;
}
