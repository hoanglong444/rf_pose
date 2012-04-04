/**
 * Author: Pierre-Luc Bacon <pierrelucbacon@aqra.ca>
 *
 * Work derived from Juergen Gall, BIWI, ETH Zurich <gall@vision.ee.ethz.ch>
 */
#include "CRTree.h"

#include <opencv/cv.h>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>

CRTree::CRTree(int minSamples, int maxDepth) : 
    minSamples(minSamples), maxDepth(maxDepth), numLeaves(0)
{
	numNodes = (int) pow(2.0, (int) (maxDepth + 1)) - 1;
	
	// num_nodes x 7 matrix as vector
	treetable = new int[numNodes * 7];
	memset(treetable, 0, sizeof(int)*numNodes*7);
	
	// allocate memory for leafs
	leaves = new LeafNode[(int) pow(2.0, int(maxDepth))];
}

CRTree::CRTree(const std::string& filename)
{
    loadTree(filename);
}
	
CRTree::~CRTree() 
{
    delete[] leaves; 
    delete[] treetable;
}

void CRTree::grow(std::vector<ImagePatch>& patches) 
{
    // Seed the RNG before generating random tests
    cv::theRNG().state = time(NULL);

    std::cout << "Growing tree ..." << std::endl;
    
    std::vector<ImagePatch*> data(patches.size());
    for (unsigned i = 0; i < patches.size(); i++) {
        data[i] = &patches[i];
    }
	grow(data, 0, 0, data.size());
}

void CRTree::grow(std::vector<ImagePatch*>& data, int node, unsigned int depth, int samples) 
{
	if(depth >= maxDepth && data.size() > 0) {	
        std::cout << "Reached maximum depth. Creating leaf" << std::endl;
		makeLeaf(data, node);
		return;
	}

    std::cout << data.size() << " samples considered at current node " << node << std::endl;
	
    int test[6];
    std::vector<ImagePatch*> partA;
    std::vector<ImagePatch*> partB;
    partA.reserve(data.size()/2);
    partB.reserve(data.size()/2);
        
    // Find optimal test
    if(optimizeTest(partA, partB, data, samples, test)) {
        // Store binary test for current node
        int* ptT = &treetable[node*7];
        ptT[0] = -1; ++ptT; 
        for(int t = 0; t < 6; ++t)
            ptT[t] = test[t];
        std::cout << "Binary test stored at current node..." << std::endl;

        // If enough patches are left recursively grow left branch
        if(partA.size() > minSamples) {
            std::cout << "Growing left branch" << std::endl;
            grow(partA, 2*node+1, depth+1, samples);
        } else {
            std::cout << "Making leaf in left branch" << std::endl;
            makeLeaf(partA, 2*node+1);
        }

        // If enough patches are left recursively grow right branch
        if(partB.size() > minSamples) {
            std::cout << "Growing right branch" << std::endl;
            grow(partB, 2*node+2, depth+1, samples);
        } else {
            std::cout << "Making leaf in right branch" << std::endl;
            makeLeaf(partB, 2*node+2);
        }
    } else {
        std::cerr << "*************** Could not find valid split. Making leaf." << std::endl;
        // Could not find split (only invalid one leave split)
        makeLeaf(data, node);
    }
	
}

bool CRTree::optimizeTest(std::vector<ImagePatch*>& partA, std::vector<ImagePatch*>& partB, std::vector<ImagePatch*>& data, unsigned iter, int* test) 
{   	        
	// Get the dimensions of a patch. They should all be of the same size.
	double width = data[0]->patch.size().width;
	double height = data[0]->patch.size().height;	
	
    std::vector<IntIndex> valSet(data.size());
       	                
    double bestSplit = -DBL_MAX;
    bool ret = false;
	    
	// Find best test
	for(unsigned i = 0; i < 10; ++i) {    
		// generate binary test for pixel locations m1 and m2
        int tmpTest[6];
		generateTest(tmpTest, width, height);

		// compute the difference between pixel itensities for each patch
		evaluateTest(data, tmpTest, valSet);

		// find min/max values of differences between m1 and m2
		int vmin = valSet.front().difference;
		int vmax = valSet.at(data.size()-1).difference;
		
		if((vmax - vmin) > 0) {
            // Find best threshold
            for(unsigned int j = 0; j < N_THRESHOLD_IT; j++) { 
                // Generate some random thresholds
                int tr = cv::theRNG().uniform(vmin, vmax);
                
                // Split training data into two sets A and B accroding to threshold 
                std::vector<ImagePatch*> tmpA;
           	    std::vector<ImagePatch*> tmpB;
           	    tmpA.reserve(data.size()/2);
           	    tmpB.reserve(data.size()/2);
           	                
                split(data, tr, valSet, tmpA, tmpB);
                               
				// Do not allow empty set split (all patches end up in set A or B)
				if((tmpA.size() > 5) && (tmpB.size() > 5)) {
					// Measure quality of split
					double score = measureInformationGain(data, tmpA, tmpB);
                    
					// Take binary test with best split
					if(score > bestSplit) {
						ret = true;
						bestSplit = score;

						memcpy(test, tmpTest, sizeof(tmpTest));
						test[5] = tr;

						partA = tmpA;
						partB = tmpB;
					}
				}
			}
		}
	}

	// return true if a valid test has been found
	// test is invalid if only splits with an empty set A or B has been created
	return ret;
}

void CRTree::generateTest(int* test, unsigned width, unsigned height) 
{
    // Location of pixel m1 in this patch
	test[0] = cv::theRNG().uniform(0, width);
	test[1] = cv::theRNG().uniform(0, height);
	
	// Location of pixel m2
	test[2] = cv::theRNG().uniform(0, width);
	test[3] = cv::theRNG().uniform(0, height);
}

void CRTree::evaluateTest(std::vector<ImagePatch*>& data, const int* test, std::vector<IntIndex>& valSet) 
{
    unsigned i = 0;
    for (auto it = data.begin(); it < data.end(); it++, i++) {
        int m1 = (*it)->patch.at<uchar>(test[1], test[0]);
        int m2 = (*it)->patch.at<uchar>(test[3], test[2]);  
        
        valSet[i] = IntIndex(m1 - m2, i);
    }
    
    std::sort(valSet.begin(), valSet.begin() + data.size());
}

void CRTree::split(std::vector<ImagePatch*>& data, int tr, std::vector<IntIndex>& valSet, std::vector<ImagePatch*>& partA, std::vector<ImagePatch*>& partB) 
{
    // Sorted on the difference m1 - m2
    std::vector<IntIndex>::iterator cutoff;
    for (cutoff = valSet.begin(); cutoff < (valSet.begin()+data.size()); cutoff++) {
        if ((*cutoff).difference > tr) {
            break;
        }
    }
        
    // IntIndex contains index back to training set (unsorted)
    for (auto it = valSet.begin(); it < cutoff; it++) {
        partA.push_back(data[(*it).index]);
    }
    
    for (auto it = cutoff; it < valSet.end(); it++) {
        partB.push_back(data[(*it).index]);
    }
}

double CRTree::measureInformationGain(std::vector<ImagePatch*>& parent, std::vector<ImagePatch*>& partA, std::vector<ImagePatch*>& partB) 
{
    // IG = \log |\Sigm a(P)| - \sum_{i \in \{L, R\}} w_i \log |\Sigma_i (P_i)|
    // w_i = \frac{|P_i|}{|P|}
    double Wl = (double)partA.size()/(double)parent.size();
    double Wr = (double)partB.size()/(double)parent.size();
    
    //std::cout << "Wl " << Wl << " Wr " << Wr << std::endl;
    
    // Compute the covariance matrices
    // Two elements: pitch, yaw
    cv::Mat P(parent.size(), 2, CV_64F);
    for (unsigned i = 0; i < parent.size(); i++) {
        P.at<double>(i, 0) = parent[i]->pitch;
        P.at<double>(i, 1) = parent[i]->yaw;        
    }
    cv::Mat covP(0, 0, CV_64F);
    cv::Mat meanP(0, 0, CV_64F);
    cv::calcCovarMatrix(P, covP, meanP, CV_COVAR_ROWS | CV_COVAR_NORMAL | CV_COVAR_SCALE);    
    
    // Left branch
    cv::Mat Pl(partA.size(), 2, CV_64F);
    for (unsigned i = 0; i < partA.size(); i++) {
        Pl.at<double>(i, 0) = partA[i]->pitch;
        Pl.at<double>(i, 1) = partA[i]->yaw;        
    }
    cv::Mat covPl(0, 0, CV_64F);
    cv::Mat meanPl(0, 0, CV_64F);
    cv::calcCovarMatrix(Pl, covPl, meanPl, CV_COVAR_ROWS | CV_COVAR_NORMAL | CV_COVAR_SCALE);        
    
    // Right branch
    cv::Mat Pr(partB.size(), 2, CV_64F);
    for (unsigned i = 0; i < partB.size(); i++) {
        Pr.at<double>(i, 0) = partB[i]->pitch;
        Pr.at<double>(i, 1) = partB[i]->yaw;        
    }
    cv::Mat covPr(0, 0, CV_64F);
    cv::Mat meanPr(0, 0, CV_64F);
    cv::calcCovarMatrix(Pr, covPr, meanPr, CV_COVAR_ROWS | CV_COVAR_NORMAL | CV_COVAR_SCALE);        

    double ig = log(cv::determinant(covP)) - Wr*log(cv::determinant(covPr)) - Wl*log(cv::determinant(covPl));
    
    if (isinf(ig)) {
        ig = 0; 
    }
    return ig;
}

// Create leaf node from patches 
void CRTree::makeLeaf(std::vector<ImagePatch*>& data, int node) {
    std::cout << "Making leaf " << numLeaves << " with " << data.size() << " samples." << std::endl;
            
	// Get pointer
	treetable[node*7] = numLeaves;
	LeafNode* leaf = &leaves[numLeaves];

	// Store sigma and mu
	if (data.size() > 0) {
        cv::Mat P(data.size(), 2, CV_64F);
        for (unsigned i = 0; i < data.size(); i++) {
            P.at<double>(i, 0) = data[i]->pitch;
            P.at<double>(i, 1) = data[i]->yaw;        
        }
        cv::calcCovarMatrix(P, leaf->cov, leaf->mean, CV_COVAR_ROWS | CV_COVAR_NORMAL | CV_COVAR_SCALE);   
    }
    
	// Increase leaf counter
    numLeaves += 1;
}

bool CRTree::loadTree(const std::string& filename)
{
	int dummy;
    std::ifstream file(filename);
	if(file.is_open()) {
		// allocate memory for tree table
		file >> maxDepth;
		numNodes = (int)pow(2.0,int(maxDepth+1))-1;

		// num_nodes x 7 matrix as vector
		treetable = new int[numNodes * 7];
		int* ptT = &treetable[0];
		
		// allocate memory for leafs
		file >> numLeaves;
		leaves = new LeafNode[numLeaves];

		// read tree nodes
		for(unsigned int n = 0; n < numNodes; ++n) {
			file >> dummy; file >> dummy;
			for(unsigned int i = 0; i < 7; ++i, ++ptT) {
				file >> *ptT;
			}
		}

		// read tree leafs
		LeafNode* ptLN = &leaves[0];
		for(unsigned int l = 0; l < numLeaves; ++l, ++ptLN) {
			file >> dummy;

            cv::Mat meanPr(1, 2, CV_64F);
			file >> meanPr.at<double>(0, 0);
            file >> meanPr.at<double>(0, 1);
			
            cv::Mat covPr(2, 2, CV_64F);
            file >> covPr.at<double>(0, 0);
            file >> covPr.at<double>(0, 1);
            file >> covPr.at<double>(1, 0);
            file >> covPr.at<double>(1, 1);

            ptLN->mean = meanPr;
            ptLN->cov = covPr;
		}

	} else {
        std::cerr << "Could not read tree: " << filename << std::endl;
        return false;
	}

	file.close();
    return true;
}

bool CRTree::saveTree() const
{
  time_t rawtime;
  time (&rawtime);
  struct tm* timeinfo;
  timeinfo = localtime (&rawtime);

  char buffer[80];
  strftime (buffer, sizeof(buffer), "%m-%d-%y-%X", timeinfo);

  return saveTree(std::string("random-tree-") + buffer); 
}

bool CRTree::saveTree(const std::string& filename) const 
{
	std::cout << "Saving tree to file: " << filename << " ... "  << std::endl;
    
	bool done = false;

	std::ofstream out(filename);
	if(out.is_open()) {
        // Write header: max depth & number of leaves
		out << maxDepth << " " << numLeaves << std::endl;

        // Write internal nodes
		int* ptT = &treetable[0];
		int depth = 0;
		unsigned int step = 2;
		for(unsigned int n = 0; n < numNodes; n++) {
			// Compute depth of node n 
			if(n == step-1) {
				++depth;
				step *= 2;
			}

            // Write node information and associated test on a line
			out << n << " " << depth << " ";
			for(unsigned int i = 0; i < 7; ++i, ++ptT) {
				out << *ptT << " ";
			}
			out << std::endl;
		}
		out << std::endl;

        // Write leaves
		LeafNode* ptLN = &leaves[0];
		for(unsigned int l = 0; l < numLeaves; ++l, ++ptLN) {
			out << l << std::fixed << " " 
                << ptLN->mean.at<double>(0, 0) << " " << ptLN->mean.at<double>(0, 1)
                << " " << ptLN->cov.at<double>(0, 0) << " " << ptLN->cov.at<double>(0, 1) 
                << " " << ptLN->cov.at<double>(1, 0) << " " << ptLN->cov.at<double>(1, 1) << std::endl;
		}
		out.close();
		done = true;
	}

	return done;
}

const LeafNode* CRTree::regression(const cv::Mat& patch) const {
	int node = 0;
	const int* pnode = &treetable[0];

	// Go through tree until one arrives at a leaf. -1 means internal node
	while(pnode[0]==-1) {
        int m1 = patch.at<uchar>(pnode[2], pnode[1]);
        int m2 = patch.at<uchar>(pnode[4], pnode[3]);  

		bool test = (m1 - m2) >= pnode[6];

        // 2n + 1 : Left
        // 2n + 2 : Right
		int incr = node + 1 + test;
		node += incr;
		pnode += incr * 7;
	}

	return &leaves[pnode[0]];
}
