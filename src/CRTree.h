/* 
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#pragma once

#define sprintf_s sprintf 

#include "CRPatch.h"
#include <iostream>
#include <fstream>

// Auxilary structure
struct IntIndex {
	int val;
	unsigned int index;
	bool operator<(const IntIndex& a) const { return val<a.val; }
};

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
	void grow(const std::vector<std::vector<const PatchFeature*> >& TrainSet, int node, unsigned int depth, int samples, float pnratio);
	
	void makeLeaf(const std::vector<std::vector<const PatchFeature*> >& TrainSet, float pnratio, int node);
	
	bool optimizeTest(std::vector<std::vector<const PatchFeature*> >& SetA, std::vector<std::vector<const PatchFeature*> >& SetB, const std::vector<std::vector<const PatchFeature*> >& TrainSet, int* test, unsigned int iter, unsigned int mode);
	
	void generateTest(int* test, unsigned int max_w, unsigned int max_h, unsigned int max_c);
	
	void evaluateTest(std::vector<std::vector<IntIndex> >& valSet, const int* test, const std::vector<std::vector<const PatchFeature*> >& TrainSet);
	
	void split(std::vector<std::vector<const PatchFeature*> >& SetA, std::vector<std::vector<const PatchFeature*> >& SetB, const std::vector<std::vector<const PatchFeature*> >& TrainSet, const std::vector<std::vector<IntIndex> >& valSet, int t);
	
	double measureInformationGain(const std::vector<std::vector<const PatchFeature*> >& SetA, const std::vector<std::vector<const PatchFeature*> >& SetB);

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

