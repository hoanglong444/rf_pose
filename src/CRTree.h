/* 
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#include "FeatureExtractor.h"

// Structure for the leafs
struct LeafNode {
	// Constructors
	LeafNode() {}
    // Empirical covariance matrix	
	cv::Mat cov;
    // Empirical mean	
	cv::Mat mean;
};

class CRTree {
public:
	/**
	 * Creates a new tree
	 * @param minSamples Minimum number of samples. A termination condition
	 * @param maxDepth Maximum tree depth during training.
	 */
	CRTree(int minSamples=20, int maxDepth=15);

    /**
     * Loads an actual serialized tree stored in a file.
     * @param filename Path to the file
     */
    CRTree(const std::string& filename);
	
	~CRTree();
	
	/**
	 * Drop a patch down the tree and retreive the final leaf
     * @param patch An image patch to evaluate
     * @return A leaf node object containing the parameters for the distribution of interest 
	 */
    const LeafNode* regression(const ImagePatch& patch) const; 

    /**
     * Train tree using n samples from training set.
     * @param training The training data
     * @param n The number of samples
     */
	void grow(std::vector<ImagePatch>& patches);
	
	/**
	 * Save the tree table to a file
	 * @param filename The output filename
	 */
	bool saveTree(const std::string& filename) const;
	
	// Number of iterations for optimizing the threshold
	static constexpr unsigned N_THRESHOLD_IT = 10;

private:
    // Produced by the evaluateTest() function
    struct IntIndex {
        IntIndex(int difference=0, unsigned int index=0) : difference(difference), index(index) {};
   	    bool operator<(const IntIndex& a) const { return difference < a.difference; }
	    int difference;
    	unsigned int index;
    };
    
    typedef std::vector<ImagePatch> TrainingSet;   
			
	void grow(std::vector<ImagePatch*>& data, int node, unsigned int depth, int samples);
	
	bool optimizeTest(std::vector<ImagePatch*>& partA, std::vector<ImagePatch*>& partB, std::vector<ImagePatch*>& data, unsigned iter, int* test);
	
	void generateTest(int* test, unsigned width, unsigned height);	
	
	void evaluateTest(std::vector<ImagePatch*>& data, const int* test, std::vector<IntIndex>& valSet);
	
	void split(std::vector<ImagePatch*>& data, int tr, std::vector<IntIndex>& valSet, std::vector<ImagePatch*>& partA, std::vector<ImagePatch*>& partB);	
	
	double measureInformationGain(std::vector<ImagePatch*>& parent, std::vector<ImagePatch*>& partA, std::vector<ImagePatch*>& partB);	
	
	void makeLeaf(std::vector<ImagePatch*>& data, int node);

    bool loadTree(const std::string& filename);
	
	// Data structure
	// tree table
	// 2^(max_depth+1)-1 x 7 matrix as vector
	// column: leafindex x1 y1 x2 y2 channel thres
	// if node is not a leaf, leaf=-1
	int* treetable;

	//leafs as vector
	LeafNode* leaves;

	// stop growing when number of patches is less than min_samples
	unsigned int minSamples;

	// depth of the tree: 0-max_depth
	unsigned int maxDepth;

	// number of nodes: 2^(max_depth+1)-1
	unsigned int numNodes;

	// number of leafs
	unsigned int numLeaves;
};

