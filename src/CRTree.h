/* 
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#include "FeatureExtractor.h"

// Structure for the leafs
struct LeafNode {
	// Constructors
	LeafNode() {}
	
	// Probability of belonging to a head
	float pfg;
	
	// mean vector
	cv::Mat mean;
	
	// trace of the covariance matrix
	float trace;
};

class CRTree {
public:
	/**
	 * Creates a new tree
	 * @param minSamples Minimum number of samples. A termination condition
	 * @param maxDepth Maximum tree depth during training.
	 */
	CRTree(int minSamples=20, int maxDepth=15);
	
	~CRTree();
	
	/**
	 * Given a patch, compute the estimated pose.
	 */
	//const LeafNode* regression(uchar** ptFCh, int stepImg) const;

    /**
     * Train tree using n samples from training set.
     * @param training The training data
     * @param n The number of samples
     */
	void grow(const std::vector<ImagePatch>& patches);
	
	// Number of iterations for optimizing the threshold
	static constexpr unsigned N_THRESHOLD_IT = 10;

private:
    // Produced by the evaluateTest() function
    struct IntIndex {
        IntIndex(int difference, unsigned int index) : difference(difference), index(index) {};
   	    bool operator<(const IntIndex& a) const { return difference < a.difference; }
	    int difference;
    	unsigned int index;
    };
    
    typedef std::vector<ImagePatch> TrainingSet;   
			
	void grow(const TrainingSet& data, int node, unsigned int depth, int samples);
	
	bool optimizeTest(TrainingSet& partA, TrainingSet& partB, const TrainingSet& data, unsigned iter, int* test);
	
	void generateTest(int* test, unsigned width, unsigned height);	
	
	void evaluateTest(const TrainingSet& data, const int* test, std::vector<IntIndex>& valSet);
	
	void split(const TrainingSet& data, int tr, std::vector<IntIndex>& valSet, TrainingSet& partA, TrainingSet& partB);	
	
	double measureInformationGain(const TrainingSet& parent, const TrainingSet& partA, const TrainingSet& partB);	
	
	void makeLeaf(const TrainingSet& data, int node);
	
	// Data structure
	// tree table
	// 2^(max_depth+1)-1 x 7 matrix as vector
	// column: leafindex x1 y1 x2 y2 channel thres
	// if node is not a leaf, leaf=-1
	int* treetable;

	// stop growing when number of patches is less than min_samples
	unsigned int minSamples;

	// depth of the tree: 0-max_depth
	unsigned int maxDepth;

	// number of nodes: 2^(max_depth+1)-1
	unsigned int numNodes;

	// number of leafs
	unsigned int numLeaves;

	//leafs as vector
	LeafNode* leaf;
};

