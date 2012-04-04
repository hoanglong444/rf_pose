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

std::tuple<double, double> PoseEstimator::estimate(const std::string& filename, double maxVariance)
{
    const cv::Mat input = cv::imread(filename, 0);
    return estimate(input);
}

std::tuple<double, double> PoseEstimator::estimate(const cv::Mat& img, double maxVariance)
{
    std::vector<cv::Mat> imagePatches;
    extractPatches(img, imagePatches);

    cv::Mat combinedMean = cv::Mat::zeros(1, 2, CV_64F);
    cv::Mat combinedCov = cv::Mat::zeros(2, 2, CV_64F);

    size_t numberVotes = 0;
    for (auto patch : imagePatches) {
        std::vector<const LeafNode*> leaves;
        forest.regression(patch, leaves);    

        // Combine the Gaussians from each leaf
        for (auto leaf : leaves) {
            //if (cv::determinant(leaf->cov) > maxVariance) {
            //    continue;
            //}
            //std::cout << "Determinant " << cv::determinant(leaf->cov) << std::endl;
            combinedMean += leaf->mean;
            combinedCov += leaf->cov;
            numberVotes += 1;
        }
    }

    std::cout << "Combined mean " << combinedMean/numberVotes << std::endl;
    std::cout << "Number of votes " << numberVotes << std::endl;

    return std::tuple<double, double>(0.0, 0.0);
}
