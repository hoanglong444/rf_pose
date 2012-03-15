#ifndef __DATASET_LOADER_H__
#define __DATASET_LOADER_H__

#include "FeatureExtractor.h"

#include <string>
#include <vector>

/**
 * This class is meant to load the 
 * images from the Pointing'04 database.
 */
class DatasetLoader 
{
public:
    DatasetLoader(const std::string& dirName);
    ~DatasetLoader();
    
    std::vector<std::string> getRandomInstances(unsigned n);
    
    /**
     * Extract patches from a random subset of the dataset
     * and store the patch representation of every image.
     * @postcondition n patch representations will be stored
     * in memory until freed by the descructor or overwritten 
     * with a subsequent call.
     */
    void processRandomSubset(unsigned n);
        
private:
    /**
     * Parse the filename to extract the pitch and yaw values.
     */
    std::pair<double, double> parsePitchYaw(const std::string& filename);
    
    std::vector<std::string> filenames;
    std::vector<ImagePatchRepresentation> processedImages;
        
    FeatureExtractor extractor;
};

#endif
