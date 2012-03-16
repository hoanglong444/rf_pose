#ifndef __FEATURE_EXTRACTOR_H__
#define __FEATURE_EXTRACTOR_H__

#include <opencv/cv.h>
#include <vector>
#include <utility>
#include <stdlib.h>
#include <time.h>
#include <iostream>
/**
 * This class contains the patches 
 * extracted by the FeatureExtrator 
 * for a given image. 
 */
struct ImagePatchRepresentation {
    ImagePatchRepresentation() : yaw(0), pitch(0) {};
    
    // Smoothed image
    cv::Mat image;
    
    // References within the smoothed image
    std::vector<cv::Mat> patches;
    
    // Centers for the patches
    std::vector<cv::Point> centers;
    
    // Orientation
    double yaw;
    
    double pitch;
};

/**
 * This class contains the information 
 * of a patch from a given image.
 */
struct ImagePatch 
{  
    ImagePatch(cv::Mat const* patch, double pitch, double yaw) : patch(patch), pitch(pitch), yaw(yaw) {};
    cv::Mat const* patch;
    double pitch;
    double yaw;
};

/**
 * Given an image, this class compute its patch representation.
 */
class FeatureExtractor {
public:
    /**
     * @param size The window size around a keypoint
     */
	FeatureExtractor(unsigned width=32, unsigned height=32);

    /**
     * Extract features from image patches around the detected keypoints
     * @param file Path to file
     */
    ImagePatchRepresentation extractPatches(const std::string& file, double yaw = 0, double pitch = 0);
    
    /**
     * Extract features from image patches around the detected keypoints
     * @param img A reference to the input image
     */
    ImagePatchRepresentation extractPatches(const cv::Mat& img, double yaw = 0, double pitch = 0);
	
protected:
    /**
     * Extract the keypoints from the image
     * @param img A pointer to the input image
     */
    std::vector<cv::KeyPoint> extractKeypoints(const cv::Mat& img);
    
private:
    // Window width
	unsigned width;
	
	// Window height
    unsigned height;
};

#endif
