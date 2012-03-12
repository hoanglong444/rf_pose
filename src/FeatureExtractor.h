#include <opencv/cv.h>
#include <vector>
#include <utility>
#include <stdlib.h>
#include <time.h>
#include <iostream>
/**
 * This class contains the patches 
 * extracted by the FeatureExtrator class. 
 */
struct PatchRepresentation {
    PatchRepresentation() {
        cv::theRNG().state = time(NULL);
    }
    
    // Smoothed image
    cv::Mat image;
    
    // References within the smoothed image
    std::vector<cv::Mat> patches;
    
    // Centers for the patches
    std::vector<cv::Point> centers;
    
    /**
     * Given a patch p, draw two pixel locations m1 and m2 at random
     * @param p The patch index
     * @return A pair of pixel intensities <m1, m2> from patch p
     */
    std::pair<uchar, uchar> getRandomPair(int p) {
        cv::Mat_<uchar> randPoints(2, 2);       
        cv::randu(randPoints, cv::Scalar(0), cv::Scalar(32)); 
        std::cout << randPoints << std::endl;
        
        return std::make_pair(patches[p].at<uchar>(0,0),
                              patches[p].at<uchar>(0,0));
    };
    
private:
    cv::RNG rng;
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
    PatchRepresentation extractPatches(const std::string& file);
    
    /**
     * Extract features from image patches around the detected keypoints
     * @param img A reference to the input image
     */
    PatchRepresentation extractPatches(const cv::Mat& img);
	
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

