#include "FeatureExtractor.h"

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <iostream>

int main(void)
{
FeatureExtractor featExtractor;

ImagePatchRepresentation patch = featExtractor.extractPatches("input.jpg");

return 0;
}
