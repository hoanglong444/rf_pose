#include "DatasetLoader.h"
#include <opencv/cv.h>

#include <sstream>
#include <algorithm>

#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>


DatasetLoader::DatasetLoader(const std::string& filename, unsigned width, unsigned height) : extractor(width, height) 
{
    cv::theRNG().state = time(NULL);

    for (unsigned i = 1; i <= POINTING04_N_PEOPLE; i++) {
        std::stringstream ss;
        ss << filename + "/" + POINTING04_PREFIX << std::setfill('0') << std::setw(2) << i;
        std::string path = ss.str();
        
        DIR* dirp = opendir(path.c_str());
    
        struct dirent* dp;
        while ((dp = readdir(dirp)) != NULL) {
            // Skip over directories
            std::string fullpath = std::string(path + "/" + dp->d_name);
            struct stat fileStat;
            lstat(fullpath.c_str(), &fileStat);
            if(S_ISDIR(fileStat.st_mode)) {
             continue;
            }
         
            filenames.push_back(fullpath);
        }
    
        closedir(dirp);
    }
}

size_t DatasetLoader::getNumberImages()
{
    return filenames.size();
}

std::vector<std::string> DatasetLoader::getRandomInstances(unsigned n)
{
    cv::Mat instancesIdx(n, 1, CV_32S);
    cv::randu(instancesIdx, 0, filenames.size());
    std::vector<std::string> chosenFiles;
    
    for (unsigned i = 0; i < n; i++) {
        chosenFiles.push_back(filenames[instancesIdx.at<uint>(i)]);
    }
    
    return chosenFiles;
}

std::pair<double, double> DatasetLoader::parsePitchYaw(const std::string& filename)
{
    size_t pos = filename.find_first_of("+-");
    std::string angles = filename.substr(pos);
    angles = angles.substr(0, angles.find(".jpg"));
   
    size_t sep = angles.find_last_of("+-");
    std::string pitchStr = angles.substr(0, sep);
    std::string yawStr = angles.substr(sep);
       
    double pitch, yaw;
    std::istringstream(pitchStr) >> pitch;
    std::istringstream(yawStr) >> yaw;

    return std::make_pair(pitch, yaw);
}

void DatasetLoader::processRandomImageSubset()
{
    processRandomImageSubset(filenames.size());
}

void DatasetLoader::processRandomImageSubset(unsigned n)
{
    auto filenames = getRandomInstances(n);
    for (auto it = filenames.begin(); it < filenames.end(); it++) {
        auto angles = parsePitchYaw(*it);
        
        std::cout << "Opening " << *it << " ... " << std::endl;
        auto imageRepresentation = extractor.extractPatches(*it, angles.first, angles.second);
        processedImages.push_back(imageRepresentation);
        
        std::cout << imageRepresentation.patches.size() << " patches extracted. " << std::endl;
        
        for (auto itPatches = imageRepresentation.patches.begin(); itPatches < imageRepresentation.patches.end(); itPatches++) {
            patches.push_back(ImagePatch((*itPatches), imageRepresentation.pitch, imageRepresentation.yaw));
        }
    }
    
    std::random_shuffle(patches.begin(), patches.end());
}

const std::vector<ImagePatchRepresentation>& DatasetLoader::getProcessedImages()
{
    return processedImages;
}

const std::vector<ImagePatch>& DatasetLoader::getPatches()
{
    return patches;
}
     
DatasetLoader::~DatasetLoader()
{

}
