#include "DatasetLoader.h"
#include "CRTree.h"
#include "CRForest.h"

#include <iostream>

int main(void)
{
    DatasetLoader loader("training");
    size_t numberImages = loader.getNumberImages();
    std::cout << numberImages << " images available." << std::endl;
    
//    loader.processRandomImageSubset(100);
//    auto patches = loader.getPatches();
    
//    CRTree tree;
//    tree.grow(patches);
    
//    tree.saveTree();

//    CRTree tree2("testTrain.txt");
//    tree2.saveTree("testTrain2.txt");
//
    CRForest forest("trees");

    return 0;
}
