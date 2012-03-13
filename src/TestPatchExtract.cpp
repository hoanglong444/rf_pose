#include "FeatureExtractor.h"

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <iostream>

int main(void)
{
FeatureExtractor featExtractor;

PatchRepresentation patch = featExtractor.extractPatches("input.jpg");

auto pair = patch.getRandomIntensityPair(1);

std::cout << "Intensity " << (int) pair.first << " and " << (int) pair.second << std::endl;

return 0;
}
