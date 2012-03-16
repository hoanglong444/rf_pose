/* 
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/
#include "CRTree.h"
#include <highgui.h>
#include <algorithm>

CRTree::CRTree(int minSamples, int maxDepth) : 
    minSamples(minSamples), maxDepth(maxDepth), numLeaves(0)
{
	numNodes = (int) pow(2.0, (int) (maxDepth + 1)) - 1;
	
	// num_nodes x 7 matrix as vector
	treetable = new int[numNodes * 7];
	memset(treetable, 0, sizeof(int)*numNodes*7);
	
	// allocate memory for leafs
	leaf = new LeafNode[(int) pow(2.0, int(maxDepth))];
}
	
CRTree::~CRTree() 
{
    delete[] leaf; 
    delete[] treetable;
}

void CRTree::grow(const std::vector<ImagePatch>& patches) 
{
    // Seed the RNG before generating random tests
    cv::theRNG().state = time(NULL);

    std::cout << "Growing tree ..." << std::endl;
    
    std::cout << patches[0].patch << std::endl;
        
	grow(patches, 0, 0, patches.size());
}

void CRTree::grow(const TrainingSet& data, int node, unsigned int depth, int samples) 
{
	if(depth >= maxDepth) {	
		makeLeaf(data, node);
		return;
	}

    TrainingSet partA;
    TrainingSet partB;
    int test[6];

    // Find optimal test
    if(optimizeTest(partA, partB, data, samples, test)) {
        // Store binary test for current node
        int* ptT = &treetable[node*7];
        ptT[0] = -1; ++ptT; 
        for(int t = 0; t < 6; ++t)
            ptT[t] = test[t];

        // If enough patches are left recursively grow left branch
        if(partA.size() > minSamples) {
            grow(partA, 2*node+1, depth+1, samples);
        } else {
            makeLeaf(partA, 2*node+1);
        }

        // If enough patches are left recursively grow right branch
        if(partB.size() > minSamples) {
            grow(partB, 2*node+2, depth+1, samples);
        } else {
            makeLeaf(partB, 2*node+2);
        }
    } else {
        // Could not find split (only invalid one leave split)
        makeLeaf(data, node);
    }
	
}

bool CRTree::optimizeTest(TrainingSet& partA, TrainingSet& partB, const TrainingSet& data, unsigned iter, int* test) 
{        
	std::cout << "Processing patch " << data[0].pitch << " " << data[0].yaw << std::endl;
	        
	// Get the dim of a patch. They should all be of the same size.
	double width = data[0].patch.size().width;
	double height = data[0].patch.size().height;	
	
    double bestSplit = -DBL_MAX;
    bool ret = false;
	
	//std::cout << "Trying to find one of " << iter << " best random test" << std::endl;
	
	// Find best test
	for(unsigned i = 0; i < iter; ++i) {
	    std::cout << "Evaluating random test " << i << std::endl;
	    
		// generate binary test for pixel locations m1 and m2
        int tmpTest[6];
		generateTest(tmpTest, width, height);

		// compute value for each patch
        // temporary data for finding best test
        std::vector<IntIndex> valSet;
		evaluateTest(data, tmpTest, valSet);

		// find min/max values of differences between m1 and m2
		int vmin = valSet.front().difference;
		int vmax = valSet.back().difference;
		
		//std::cout << "Difference " << vmax << " - " << vmin << " = " << (vmax - vmin) << std::endl;
		if((vmax - vmin) > 0) {
            // Find best threshold
            for(unsigned int j = 0; j < N_THRESHOLD_IT; j++) { 
                // Generate some random thresholds
                int tr = cv::theRNG().uniform(vmin, vmax);
                //std::cout << "Random threshold " << tr << std::endl;
                
                // Split training data into two sets A and B accroding to threshold 
                TrainingSet tmpA;
   	            TrainingSet tmpB;
                split(data, tr, valSet, tmpA, tmpB);
                
                //std::cout << "Split A " << tmpA.size() << " split B " << tmpB.size() << std::endl;
                
				// Do not allow empty set split (all patches end up in set A or B)
				if((tmpA.size() > 5) && (tmpB.size() > 5)) {
					// Measure quality of split
					double score = measureInformationGain(data, tmpA, tmpB);
                    //std::cout << "Information gain " << score << std::endl;
                    
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
			
			std::cout << "Best IG after threshold search " << bestSplit << std::endl;
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

void CRTree::evaluateTest(const TrainingSet& data, const int* test, std::vector<IntIndex>& valSet) 
{
    unsigned i = 0;
    for (auto it = data.begin(); it < data.end(); it++, i++) {
        int m1 = (*it).patch.at<uchar>(test[1], test[0]);
        int m2 = (*it).patch.at<uchar>(test[3], test[2]);  
        
        valSet.push_back(IntIndex(m1 - m2, i));    
    }
    
    std::sort(valSet.begin(), valSet.end());
}

void CRTree::split(const TrainingSet& data, int tr, std::vector<IntIndex>& valSet, TrainingSet& partA, TrainingSet& partB) 
{
    // Sorted on the difference m1 - m2
    //auto cutoff = std::upper_bound(valSet.begin(), valSet.end(), tr,
    //    [](const int threshold, const IntIndex& a) {return a.difference < threshold;});
    std::vector<IntIndex>::iterator cutoff;
    for (cutoff = valSet.begin(); cutoff < valSet.end(); cutoff++) {
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

double CRTree::measureInformationGain(const TrainingSet& parent, const TrainingSet& partA, const TrainingSet& partB) 
{
    // IG = \log |\Sigm a(P)| - \sum_{i \in \{L, R\}} w_i \log |\Sigma_i (P_i)|
    // w_i = \frac{|P_i|}{|P|}
    double Wl = (double)partA.size()/(double)parent.size();
    double Wr = (double)partB.size()/(double)parent.size();
    
    //std::cout << "Wl " << Wl << " Wr " << Wr << std::endl;
    
    // Compute the covariance matrices
    // Two elements: pitch, yaw
    cv::Mat P(parent.size(), 2, CV_32F);
    for (unsigned i = 0; i < parent.size(); i++) {
        P.at<float>(i, 0) = parent[i].pitch;
        P.at<float>(i, 1) = parent[i].yaw;        
    }
    cv::Mat covP(0, 0, CV_32F);
    cv::Mat meanP(0, 0, CV_32F);
    cv::calcCovarMatrix(P, covP, meanP, CV_COVAR_ROWS | CV_COVAR_NORMAL | CV_COVAR_SCALE);    
    
    // Left branch
    cv::Mat Pl(partA.size(), 2, CV_32F);
    for (unsigned i = 0; i < partA.size(); i++) {
        Pl.at<float>(i, 0) = partA[i].pitch;
        Pl.at<float>(i, 1) = partA[i].yaw;        
    }
    cv::Mat covPl(0, 0, CV_32F);
    cv::Mat meanPl(0, 0, CV_32F);
    cv::calcCovarMatrix(Pl, covPl, meanPl, CV_COVAR_ROWS | CV_COVAR_NORMAL | CV_COVAR_SCALE);        
    
    // Right branch
    cv::Mat Pr(partB.size(), 2, CV_32F);
    for (unsigned i = 0; i < partB.size(); i++) {
        Pr.at<float>(i, 0) = partB[i].pitch;
        Pr.at<float>(i, 1) = partB[i].yaw;        
    }
    cv::Mat covPr(0, 0, CV_32F);
    cv::Mat meanPr(0, 0, CV_32F);
    cv::calcCovarMatrix(Pr, covPr, meanPr, CV_COVAR_ROWS | CV_COVAR_NORMAL | CV_COVAR_SCALE);        

    double ig = log(cv::determinant(covP)) - Wr*log(cv::determinant(covPr)) - Wl*log(cv::determinant(covPl));
    
    if (isinf(ig)) {
        ig = 0; 
    }
    return ig;
}

// Create leaf node from patches 
void CRTree::makeLeaf(const TrainingSet& data, int node) {
	// Get pointer
	treetable[node*7] = numLeaves;
	//LeafNode* ptL = &leaf[numLeaves];

	// Store sigma and mu
    // TODO implement here
    
	// Increase leaf counter
    numLeaves += 1;
}

