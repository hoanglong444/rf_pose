/**
 * Pierre-Luc Bacon <plbacon@cim.mcgill.ca>
 */
#include "FeatureExtractor.h"
#include <vector>

// Forward declaration
class CRTree;
class LeafNode;

/**
 * This class loads trees from a directory and collects estimates from 
 * every trees when a sample must be evaluated.
 */
class CRForest 
{
public:
    /**
     * Load a forest from a diretory containing serialized trees
     * @param dir The path to a directory containing the trees
     */
	CRForest(const std::string& dir);
	
	~CRForest();

    /**
     * Drop a patch in every tree return the estimates at the leaves 
     * @param 
     */
    void regression(const cv::Mat& patch, std::vector<const LeafNode*>& outLeaves) const;

private:
	std::vector<CRTree*> trees;
};

