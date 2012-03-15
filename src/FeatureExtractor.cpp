#include "FeatureExtractor.h"
#include <opencv/highgui.h>
#include <iostream>
#include <algorithm>

FeatureExtractor::FeatureExtractor(unsigned width, unsigned height) : 
    width(width), 
    height(height)
{
}
	
ImagePatchRepresentation FeatureExtractor::extractPatches(const std::string& file, double yaw, double pitch)
{
    const cv::Mat input = cv::imread(file, 0);
    return extractPatches(input, yaw, pitch);
}
   
ImagePatchRepresentation FeatureExtractor::extractPatches(const cv::Mat& img, double yaw, double pitch) 
{
    // Extract keypoints
	std::vector<cv::KeyPoint> points = extractKeypoints(img);
   
    // Smooth the image for more robustness against
    // noise when later comparing pixel intensities   
    ImagePatchRepresentation imagePatches;
    cv::GaussianBlur(img, imagePatches.image, cv::Size(7, 7), 2, 2); 
    
    // Extract a ROI around the keypoints        
   for (auto it = points.begin(); it != points.end(); it++) {          
         int kx = it->pt.x;
         int ky = it->pt.y;
                 
         int x1 = std::max(0, (int) (kx - ((int) (width/2))));
         int x2 = std::min(img.size().width, (int) (kx + ((int) (width/2))));
         
         int y1 = std::max(0, (int) (ky - ((int) (height/2))));
         int y2 = std::min(img.size().height, (int) (ky + ((int) (height/2))));
         
         // Create a reference to a 32x32 window around the detected keypoint
         imagePatches.patches.push_back(imagePatches.image(cv::Range(y1, y2), cv::Range(x1, x2)));
         
         // Record the keypoint location
         imagePatches.centers.push_back(cv::Point(kx, ky));       
   }
   
   imagePatches.yaw = yaw;
   imagePatches.pitch = pitch;
    	
   return imagePatches;
}

std::vector<cv::KeyPoint> FeatureExtractor::extractKeypoints(const cv::Mat& img)
{
    cv::SurfFeatureDetector detector;
    
    std::vector<cv::KeyPoint> keypoints;
    detector.detect(img, keypoints);

    return keypoints;
}
