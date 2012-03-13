/* 
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/
#include "CRTree.h"
#include <highgui.h>
#include <algorithm>

CRTree::CRTree(int min_s, int max_d, int cp, CvRNG* pRNG) : 
    min_samples(min_s), max_depth(max_d), num_leaf(0), num_cp(cp), cvRNG(pRNG) 
{
	num_nodes = (int) pow(2.0, int(max_depth+1))-1;
	
	// num_nodes x 7 matrix as vector
	treetable = new int[num_nodes * 7];
	
	for(unsigned int i=0; i<num_nodes * 7; ++i) {
	    treetable[i] = 0;
	}
	
	// allocate memory for leafs
	leaf = new LeafNode[(int) pow(2.0, int(max_depth))];
}
	
CRTree::~CRTree() 
{
    delete[] leaf; 
    delete[] treetable;
}

void CRTree::growTree(const CRPatch& TrData, int samples) {
	// Get ratio positive patches/negative patches
	int pos = 0;
	vector<vector<const PatchFeature*> > TrainSet( TrData.vLPatches.size() );
	
	for(unsigned int l=0; l<TrainSet.size(); ++l) {
		TrainSet[l].resize(TrData.vLPatches[l].size());
		
		if(l>0) pos += TrainSet[l].size();
		
		for(unsigned int i=0; i<TrainSet[l].size(); ++i) {
			TrainSet[l][i] = &TrData.vLPatches[l][i];
		}
	}

	grow(TrainSet, 0, 0, samples, pos / float(TrainSet[0].size()) );
}

void CRTree::grow(const vector<vector<const PatchFeature*> >& TrainSet, int node, unsigned int depth, int samples, float pnratio) {

	if(depth<max_depth && TrainSet[1].size()>0) {	

		vector<vector<const PatchFeature*> > SetA;
		vector<vector<const PatchFeature*> > SetB;
		int test[6];

		// Find optimal test
		if( optimizeTest(SetA, SetB, TrainSet, test, samples) ) {
			// Store binary test for current node
			int* ptT = &treetable[node*7];
			ptT[0] = -1; ++ptT; 
			for(int t=0; t<6; ++t)
				ptT[t] = test[t];

			double countA = 0;
			double countB = 0;
			for(unsigned int l=0; l<TrainSet.size(); ++l) {
				cout << "Final_Split A/B " << l << " " << SetA[l].size() << " " << SetB[l].size() << endl; 
				countA += SetA[l].size(); countB += SetB[l].size();
			}
			for(unsigned int l=0; l<TrainSet.size(); ++l) {
				cout << "Final_SplitA: " << SetA[l].size()/countA << "% "; 
			}
			cout << endl;
			for(unsigned int l=0; l<TrainSet.size(); ++l) {
				cout << "Final_SplitB: " << SetB[l].size()/countB << "% "; 
			}
			cout << endl;

			// Go left
			// If enough patches are left continue growing else stop
			if(SetA[0].size()+SetA[1].size()>min_samples) {
				grow(SetA, 2*node+1, depth+1, samples, pnratio);
			} else {
				makeLeaf(SetA, pnratio, 2*node+1);
			}

			// Go right
			// If enough patches are left continue growing else stop
			if(SetB[0].size()+SetB[1].size()>min_samples) {
				grow(SetB, 2*node+2, depth+1, samples, pnratio);
			} else {
				makeLeaf(SetB, pnratio, 2*node+2);
			}

		} else {
			// Could not find split (only invalid one leave split)
			makeLeaf(TrainSet, pnratio, node);
		}	

	} else {
		// Only negative patches are left or maximum depth is reached
		makeLeaf(TrainSet, pnratio, node);
	}
}

// Create leaf node from patches 
void CRTree::makeLeaf(const std::vector<std::vector<const PatchFeature*> >& TrainSet, float pnratio, int node) {
	// Get pointer
	treetable[node*7] = num_leaf;
	LeafNode* ptL = &leaf[num_leaf];

	// Store data
	ptL->pfg = TrainSet[1].size() / float(pnratio*TrainSet[0].size()+TrainSet[1].size());
	ptL->vCenter.resize( TrainSet[1].size() );
	for(unsigned int i = 0; i<TrainSet[1].size(); ++i) {
		ptL->vCenter[i] = TrainSet[1][i]->center;
	}

	// Increase leaf counter
	++num_leaf;
}

bool CRTree::optimizeTest(vector<vector<const PatchFeature*> >& SetA, vector<vector<const PatchFeature*> >& SetB, const vector<vector<const PatchFeature*> >& TrainSet, int* test, unsigned int iter) {
	
	bool found = false;

	// temporary data for split into Set A and Set B
	vector<vector<const PatchFeature*> > tmpA(TrainSet.size());
	vector<vector<const PatchFeature*> > tmpB(TrainSet.size());

	// temporary data for finding best test
	vector<vector<IntIndex> > valSet(TrainSet.size());
	double tmpDist;
	// maximize!!!!
	double bestDist = -DBL_MAX; 
	int tmpTest[6];

	// Find best test of ITER iterations
	for(unsigned int i =0; i<iter; ++i) {

		// reset temporary data for split
		for(unsigned int l =0; l<TrainSet.size(); ++l) {
			tmpA[l].clear();
			tmpB[l].clear(); 
		}

		// generate binary test without threshold
		generateTest(&tmpTest[0], TrainSet[1][0]->roi.width, TrainSet[1][0]->roi.height, TrainSet[1][0]->vPatch.size());

		// compute value for each patch
		evaluateTest(valSet, &tmpTest[0], TrainSet);

		// find min/max values for threshold
		int vmin = INT_MAX;
		int vmax = INT_MIN;
		for(unsigned int l = 0; l<TrainSet.size(); ++l) {
			if(valSet[l].size()>0) {
				if(vmin>valSet[l].front().val)  vmin = valSet[l].front().val;
				if(vmax<valSet[l].back().val )  vmax = valSet[l].back().val;
			}
		}
		int d = vmax-vmin;

		if(d>0) {

			// Find best threshold
			for(unsigned int j=0; j<10; ++j) { 

				// Generate some random thresholds
				int tr = (cvRandInt( cvRNG ) % (d)) + vmin; 

				// Split training data into two sets A,B accroding to threshold t 
				split(tmpA, tmpB, TrainSet, valSet, tr);

				// Do not allow empty set split (all patches end up in set A or B)
				if( tmpA[0].size()+tmpA[1].size()>0 && tmpB[0].size()+tmpB[1].size()>0 ) {

					// Measure quality of split
					tmpDist = measureInformationGain(tmpA, tmpB);

					// Take binary test with best split
					if(tmpDist>bestDist) {

						found = true;
						bestDist = tmpDist;
						for(int t=0; t<5;++t) test[t] = tmpTest[t];
						test[5] = tr;
						SetA = tmpA;
						SetB = tmpB;
					}

				}

			} // end for j

		}

	} // end iter

	// return true if a valid test has been found
	// test is invalid if only splits with an empty set A or B has been created
	return found;
}


void CRTree::evaluateTest(std::vector<std::vector<IntIndex> >& valSet, const int* test, const std::vector<std::vector<const PatchFeature*> >& TrainSet) {
	for(unsigned int l=0;l<TrainSet.size();++l) {
		valSet[l].resize(TrainSet[l].size());
		for(unsigned int i=0;i<TrainSet[l].size();++i) {

			// pointer to channel
			CvMat* ptC = TrainSet[l][i]->vPatch[test[4]];
			// get pixel values 
			int p1 = (int)*(uchar*)cvPtr2D( ptC, test[1], test[0]);
			int p2 = (int)*(uchar*)cvPtr2D( ptC, test[3], test[2]);
		
			valSet[l][i].val = p1 - p2;
			valSet[l][i].index = i;			
		}
		sort( valSet[l].begin(), valSet[l].end() );
	}
}

void CRTree::split(vector<vector<const PatchFeature*> >& SetA, vector<vector<const PatchFeature*> >& SetB, const vector<vector<const PatchFeature*> >& TrainSet, const vector<vector<IntIndex> >& valSet, int t) {
	for(unsigned int l = 0; l<TrainSet.size(); ++l) {
		// search largest value such that val<t 
		vector<IntIndex>::const_iterator it = valSet[l].begin();
		while(it!=valSet[l].end() && it->val<t) {
			++it;
		}

		SetA[l].resize(it-valSet[l].begin());
		SetB[l].resize(TrainSet[l].size()-SetA[l].size());

		it = valSet[l].begin();
		for(unsigned int i=0; i<SetA[l].size(); ++i, ++it) {
			SetA[l][i] = TrainSet[l][it->index];
		}
		
		it = valSet[l].begin()+SetA[l].size();
		for(unsigned int i=0; i<SetB[l].size(); ++i, ++it) {
			SetB[l][i] = TrainSet[l][it->index];
		}

	}
}

double CRTree::measureInformationGain(const vector<vector<const PatchFeature*> >& SetA, const vector<vector<const PatchFeature*> >& SetB) {

	// get size of set A
	double sizeA = 0;
	for(vector<vector<const PatchFeature*> >::const_iterator it = SetA.begin(); it != SetA.end(); ++it) {
		sizeA += it->size();
	}

	// negative entropy: sum_i p_i*log(p_i)
	double n_entropyA = 0;
	for(vector<vector<const PatchFeature*> >::const_iterator it = SetA.begin(); it != SetA.end(); ++it) {
		double p = double( it->size() ) / sizeA;
		if(p>0) n_entropyA += p*log(p); 
	}

	// get size of set B
	double sizeB = 0;
	for(vector<vector<const PatchFeature*> >::const_iterator it = SetB.begin(); it != SetB.end(); ++it) {
		sizeB += it->size();
	}

	// negative entropy: sum_i p_i*log(p_i)
	double n_entropyB = 0;
	for(vector<vector<const PatchFeature*> >::const_iterator it = SetB.begin(); it != SetB.end(); ++it) {
		double p = double( it->size() ) / sizeB;
		if(p>0) n_entropyB += p*log(p); 
	}

	return (sizeA*n_entropyA+sizeB*n_entropyB)/(sizeA+sizeB); 
}

inline void CRTree::generateTest(int* test, unsigned int max_w, unsigned int max_h, unsigned int max_c) {
	test[0] = cvRandInt( cvRNG ) % max_w;
	test[1] = cvRandInt( cvRNG ) % max_h;
	test[2] = cvRandInt( cvRNG ) % max_w;
	test[3] = cvRandInt( cvRNG ) % max_h;
	test[4] = cvRandInt( cvRNG ) % max_c;
}

const LeafNode* CRTree::regression(uchar** ptFCh, int stepImg) const {
	// pointer to current node
	const int* pnode = &treetable[0];
	int node = 0;

	// Go through tree until one arrives at a leaf, i.e. pnode[0]>=0)
	while(pnode[0]==-1) {
		// binary test 0 - left, 1 - right
		// Note that x, y are changed since the patches are given as matrix and not as image 
		// p1 - p2 < t -> left is equal to (p1 - p2 >= t) == false
		
		// pointer to channel
		uchar* ptC = ptFCh[pnode[5]];
		// get pixel values 
		int p1 = *(ptC+pnode[1]+pnode[2]*stepImg);
		int p2 = *(ptC+pnode[3]+pnode[4]*stepImg);
		// test
		bool test = ( p1 - p2 ) >= pnode[6];

		// next node: 2*node_id + 1 + test
		// increment node/pointer by node_id + 1 + test
		int incr = node+1+test;
		node += incr;
		pnode += incr*7;
	}

	// return leaf
	return &leaf[pnode[0]];
}

