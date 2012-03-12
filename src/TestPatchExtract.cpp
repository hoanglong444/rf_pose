#include "FeatureExtractor.h"

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <iostream>

int main(void)
{
FeatureExtractor featExtractor;

PatchRepresentation patch = featExtractor.extractPatches("input.jpg");

auto pair = patch.getRandomPair(0);

std::cout << "Intensity " << (int) pair.first << std::endl;

return 0;
}
