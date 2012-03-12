#include "FeatureExtractor.h"
#include <opencv/highgui.h>
#include <iostream>
#include <algorithm>

FeatureExtractor::FeatureExtractor(unsigned width, unsigned height) : 
    width(width), 
    height(height)
{
}
	
void FeatureExtractor::extractPatches(const std::string& file)
{
    const cv::Mat input = cv::imread(file, 0);
    extractPatches(input);
}
   
void FeatureExtractor::extractPatches(const cv::Mat& img) 
{
    // Extract keypoints
	std::vector<cv::KeyPoint> points = extractKeypoints(img);

    std::cout << " keypoints detected " << points.size() << std::endl;
    
    // Smooth the image for more robustness against
    // noise when later comparing pixel intensities
    cv::Mat imgFiltered;     
    cv::GaussianBlur(img, imgFiltered, cv::Size(7, 7), 2, 2); 
    
    // Extract a ROI around the keypoints        
   for (auto it = points.begin(); it != points.end(); it++) {          
         int kx = it->pt.x;
         int ky = it->pt.y;
                 
         int x1 = std::max(0, (int) (kx - ((int) (width/2))));
         int x2 = std::min(img.size().width, (int) (kx + ((int) (width/2))));
         
         int y1 = std::max(0, (int) (ky - ((int) (height/2))));
         int y2 = std::min(img.size().height, (int) (ky + ((int) (height/2))));

         //std::cout << "x1 " << x1 << " x2 " << x2  << " y1 " << y1 << " y2 " << y2 << std::endl;   
               
         cv::Mat window = img(cv::Range(y1, y2), cv::Range(x1, x2));
   }
    	
 
}

std::vector<cv::KeyPoint> FeatureExtractor::extractKeypoints(const cv::Mat& img)
{
    cv::SurfFeatureDetector detector;
    
    std::vector<cv::KeyPoint> keypoints;
    detector.detect(img, keypoints);

    return keypoints;
}
