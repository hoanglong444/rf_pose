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
    DatasetLoader(const std::string& dirName, unsigned width=16, unsigned height=16);
    
    ~DatasetLoader();

    /**
     * Get the number of images that can be processed.
     * @return The number of images available. 
     */ 
    size_t getNumberImages();

    /**
     * Extract patches from a random subset of the dataset
     * and store the patch representation of every image.
     * @postcondition n patch representations will be stored
     * in memory until freed by the descructor or overwritten 
     * with a subsequent call.
     */
     void processRandomImageSubset();
     
    /**
     * Extract patches from a random subset of the dataset
     * and store the patch representation of every image.
     * @postcondition n patch representations will be stored
     * in memory until freed by the descructor or overwritten 
     * with a subsequent call.
     */
    void processRandomImageSubset(unsigned n);
    
    /**
     *
     * @precondition processRandomSubset should have been called 
     * prior to this.
     */
    const std::vector<ImagePatchRepresentation>& getProcessedImages();
    
     /**
     * @return All the patches, randomly shuffled, from the training set.
     * @precondition processRandomSubset should have been called 
     * prior to this.
     */
     const std::vector<ImagePatch>& getPatches();
     
     // There are fifteen people in the Pointing04' database
     static constexpr unsigned POINTING04_N_PEOPLE = 15;
  
     // The filenames all starts with "Personne"      
     static constexpr const char* POINTING04_PREFIX = "Personne"; 
       
private:
    /**
     * Parse the filename to extract the pitch and yaw values.
     * @param filename The filename following the Pointing'04 format
     * @return A pair pitch, yaw
     */
    std::pair<double, double> parsePitchYaw(const std::string& filename);
    
    /**
     * @param n The number of filenames to return
     * @return n filenames chosen at random from the available set
     */
    std::vector<std::string> getRandomInstances(unsigned n);
        
    // Contains all the names of the accessible files
    std::vector<std::string> filenames;
    
    // Contains the result of the patch extraction for each image
    std::vector<ImagePatchRepresentation> processedImages;
    
    // Contains all the patchs from all the images, randomly shuffled
    std::vector<ImagePatch> patches;
    
    FeatureExtractor extractor;
};

#endif
