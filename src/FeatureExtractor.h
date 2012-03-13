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
    std::pair<uchar, uchar> getRandomIntensityPair(int p) {
        cv::Mat_<uchar> randX(2, 1);   
        cv::Mat_<uchar> randY(2, 1);           
        cv::randu(randX, cv::Scalar(0), cv::Scalar(patches[p].size().width)); 
        cv::randu(randY, cv::Scalar(0), cv::Scalar(patches[p].size().height)); 
                
        //std::cout << randX << randY << centers[p] << std::endl;
        
        return std::make_pair(patches[p].at<uchar>(randX.at<uchar>(0, 0), randY.at<uchar>(0, 0)),
                              patches[p].at<uchar>(randX.at<uchar>(1, 0), randY.at<uchar>(1, 0)));
    };
    
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

