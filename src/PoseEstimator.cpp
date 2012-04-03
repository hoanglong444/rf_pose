#include "PoseEstimator.h"
#include "CRTree.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>

PoseEstimator::PoseEstimator(const CRForest& forest, unsigned width, unsigned height): 
    forest(forest),
    patchWidth(width),
    patchHeight(height)
{

}

PoseEstimator::~PoseEstimator()
{

}

void PoseEstimator::extractPatches(const cv::Mat& img, std::vector<cv::Mat>& imagePatches)
{
    // Extract keypoints
    cv::SurfFeatureDetector detector;
    std::vector<cv::KeyPoint> keypoints;
    detector.detect(img, keypoints);

    // Blur the image to remove some of the noise
    cv::Mat blurImg;
    cv::GaussianBlur(img, blurImg, cv::Size(7, 7), 2, 2); 

    // Extract image regions around keypoints
    imagePatches.reserve(keypoints.size());

    for (auto it = keypoints.begin(); it != keypoints.end(); it++) {          
        int kx = it->pt.x;
        int ky = it->pt.y;
                 
        int x1 = std::max(0, (int) (kx - ((int) (patchWidth/2))));
        int x2 = std::min(img.size().width, (int) (kx + ((int) (patchWidth/2))));
         
        int y1 = std::max(0, (int) (ky - ((int) (patchHeight/2))));
        int y2 = std::min(img.size().height, (int) (ky + ((int) (patchHeight/2))));
         
        // Create a reference to a 32x32 window around the detected keypoint
        imagePatches.push_back(blurImg(cv::Range(y1, y2), cv::Range(x1, x2)));
    }
}

std::tuple<float, float> PoseEstimator::estimate(const std::string& filename)
{
    const cv::Mat input = cv::imread(filename, 0);
    return estimate(input);
}

std::tuple<float, float> PoseEstimator::estimate(const cv::Mat& img)
{
    std::vector<cv::Mat> imagePatches;
    extractPatches(img, imagePatches);

    auto leaves = forest.regression(path);    

    return std::tuple<float, float>(0.0, 0.0);
}
