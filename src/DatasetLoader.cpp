#include "DatasetLoader.h"
#include <opencv/cv.h>

#include <sstream>

#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>


DatasetLoader::DatasetLoader(const std::string& filename)
{
    cv::theRNG().state = time(NULL);

    
    for (int i = 1; i < 16; i++) {
        std::stringstream ss;
        ss << filename + "/Personne" << std::setfill('0') << std::setw(2) << i;
        std::string path = ss.str();
        
        DIR* dirp = opendir(path.c_str());
    
        struct dirent* dp;
        while ((dp = readdir(dirp)) != NULL) {
            // Skip over directories
            struct stat fileStat;
            lstat(std::string(path + "/" + dp->d_name).c_str(), &fileStat);
            if(S_ISDIR(fileStat.st_mode)) {
             continue;
            }
         
            filenames.push_back(dp->d_name);
        }
    
        closedir(dirp);
    }
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

void DatasetLoader::processRandomSubset(unsigned n)
{
    auto filenames = getRandomInstances(n);
    for (auto it = filenames.begin(); it < filenames.end(); it++) {
        auto angles = parsePitchYaw(*it);
        processedImages.push_back(extractor.extractPatches(*it, angles.first, angles.second));
    }
    
    std::cout << "Patches " << processedImages.size() << std::endl;

}
    
DatasetLoader::~DatasetLoader()
{

}
