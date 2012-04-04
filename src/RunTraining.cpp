#include "DatasetLoader.h"
#include "CRTree.h"
#include "CRForest.h"

#include <iostream>

int main(int argc, char* argv[])
{
    if (argc <= 1) {
        std::cerr << argv[0] << " [path]" << std::endl;
        return -1;
    }

    DatasetLoader loader(argv[1]);
    size_t numberImages = loader.getNumberImages();
    std::cout << numberImages << " images available." << std::endl;
    
    loader.processRandomImageSubset((int) numberImages/2);
    auto patches = loader.getPatches();
    
    CRTree tree;
    tree.grow(patches);
    tree.saveTree();

    return 0;
}
