#include <opencv/cv.h>
#include <vector>

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
    void extractPatches(const std::string& file);
    
    /**
     * Extract features from image patches around the detected keypoints
     * @param img A reference to the input image
     */
    void extractPatches(const cv::Mat& img);
	
protected:
    /**
     * Extract the keypoints from the image
     * @param img A pointer to the input image
     */
    std::vector<cv::KeyPoint> extractKeypoints(const cv::Mat& img);
    
private:
	unsigned width;
    unsigned height;
};

