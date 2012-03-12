#include "FeatureExtractor.h"

#include <opencv/cv.h>
#include <opencv/highgui.h>

int main(void)
{
FeatureExtractor patch;

patch.extractPatches("input.jpg");

return 0;
}
