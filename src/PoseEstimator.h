#include "CRForest.h"

class PoseEstimator 
{
public:
    PoseEstimator(const CRForest& forest, unsigned width=16, unsigned height=16); 
    ~PoseEstimator();

    /** 
     * Estimate the pitch and yaw angles for a given image
     * @param image The image on which keypoints will be extracted and evaluated
     * @return A pair <pitch, yaw>
     */
    std::tuple<float, float> estimate(const cv::Mat& image);

    /** 
     * Estimate the pitch and yaw angles for a given image
     * @param filename Path to image on which keypoints will be extracted and evaluated
     * @return A pair <pitch, yaw>
     */
    std::tuple<float, float> estimate(const std::string& filename);

protected:
    /**
     * Extract image regions around the extracted keypoints in the image. 
     * @param img The image from which to extract keypoints
     * @param imagePatches A reference to a vector that will contain the patches
     */
    void extractPatches(const cv::Mat& img, std::vector<cv::Mat>& imagePatches);

private:
    const CRForest& forest;
    unsigned patchWidth;
    unsigned patchHeight;
};
