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
	 * @param maxDepth Maximum depth.
	 */
	CRTree(int minSamples, int maxDepth, int cp, CvRNG* pRNG);
	
	~CRTree();

    /**
     * Maximum training depth
     * @return the current maximum depth set for training
     */
	unsigned int GetDepth() const {return max_depth;}
	
	unsigned int GetNumCenter() const {return num_cp;}

	const LeafNode* regression(uchar** ptFCh, int stepImg) const;

    /**
     * Train tree using n samples from training set.
     * @param training The training data
     * @param n The number of samples
     */
	void growTree(const PatchRepresentation& training, int samples);

private:
    // Produced by the evaluateTest() function
    struct IntIndex {
	    int difference;
    	unsigned int index;
	    bool operator<(const IntIndex& a) const { return difference < a.difference; }
    };
    
    typedef std::vector<const ImagePatchRepresentation&> TrainingSet;
    
    typedef std::vector<std::vector<IntIndex> > EvaluatedTrainingSet;
    
	void evaluateTest(EvaluatedTrainingSet& valSet, const int* test, const TrainingSet& data);
	
	void split(TrainingSet& SetA, TrainingSet& SetB, const TrainingSet& data, const EvaluatedTrainingSet& valSet, int t);
	
	double measureInformationGain(const TrainingSet& SetA, const TrainingSet& SetB);
		
	void generateTest(int* test, unsigned int width, unsigned int height);
	
	bool optimizeTest(TrainingSet& SetA, TrainingSet& SetB, const TrainingSet& data, int* test, unsigned int iter);
	
	void grow(const TrainingSet& data, int node, unsigned int depth, int samples);
	
	void makeLeaf(const TrainingSet& data, int node);
	
	// Data structure
	// tree table
	// 2^(max_depth+1)-1 x 7 matrix as vector
	// column: leafindex x1 y1 x2 y2 channel thres
	// if node is not a leaf, leaf=-1
	int* treetable;

	// stop growing when number of patches is less than min_samples
	unsigned int min_samples;

	// depth of the tree: 0-max_depth
	unsigned int max_depth;

	// number of nodes: 2^(max_depth+1)-1
	unsigned int num_nodes;

	// number of leafs
	unsigned int num_leaf;

	//leafs as vector
	LeafNode* leaf;
};

