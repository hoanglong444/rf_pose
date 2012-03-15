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
         
            std::cout << dp->d_name << std::endl;;     
        }
    
        closedir(dirp);
    }
}

//std::string DatasetLoader::getRandomInstances(unsigned n)


DatasetLoader::~DatasetLoader()
{

}
