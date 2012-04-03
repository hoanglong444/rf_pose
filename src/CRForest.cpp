#include "CRForest.h"
#include "CRTree.h"

#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>

CRForest::CRForest(const std::string& path) 
{
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
    
       std::cout << "Loading tree " << fullpath << std::endl;     
       trees.push_back(new CRTree(fullpath));
    }
   
    closedir(dirp); 
}

CRForest::~CRForest()
{
    for(auto it = trees.begin(); it != trees.end(); ++it) {
        delete *it;
    }
    trees.clear();
}

void CRForest::regression(const ImagePatch& patch, std::vector<const LeafNode*>& outLeaves)
{
	outLeaves.resize(trees.size());

	for(unsigned i = 0; i < trees.size(); i++) {
		outLeaves[i] = trees[i]->regression(patch);
	}
}
