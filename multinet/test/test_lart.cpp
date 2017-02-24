#include "test.h"
#include <unordered_set>
#include <string>
#include "../include/multinet.h"

using namespace mlnet;

void test_lart() {

	test_begin("ML-LART");

	lart k;

//	MLNetworkSharedPtr mnet3 = read_multilayer("test/data/sample.mpx","sample",',');
	MLNetworkSharedPtr mnet3 = read_multilayer("test/data/sample.mpx","sample",',');
	//MLNetworkSharedPtr mnet3 = read_multilayer("/home/mikki/Downloads/friendfeed_ita.mpx","sample",',')

	uint32_t t = 2;
	float eps = 0.8;
	float gamma = 1;

	k.get_ml_community(mnet3, t, eps, gamma);

	/*

	matrix power unit test
	ublas::matrix<float> aa = ublas::zero_matrix<float>(2,2);

	aa(0,0) = 1;aa(0,1) = 1;
	aa(1,0) = 1;aa(1,1) = 1;

	printm(matrix_power(aa , 2));
	---

	block_diag unit test
	std::vector<ublas::matrix<float>> v;

	ublas::matrix<float> aa = ublas::zero_matrix<float>(3,3);
	ublas::matrix<float> bb = ublas::zero_matrix<float>(3,3);
	ublas::matrix<float> cc = ublas::zero_matrix<float>(3,3);

	aa(0,0) = 1;aa(0,1) = 1;aa(0,2) = 1;
	aa(1,0) = 1;aa(1,1) = 1;aa(1,2) = 1;
	aa(2,0) = 1;aa(2,1) = 1;aa(2,2) = 1;

	bb(0,0) = 2;bb(0,1) = 2;bb(0,2) = 2;
	bb(1,0) = 2;bb(1,1) = 2;bb(1,2) = 2;
	bb(2,0) = 2;bb(2,1) = 2;bb(2,2) = 2;

	cc(0,0) = 3;cc(0,1) = 3;cc(0,2) = 3;
	cc(1,0) = 3;cc(1,1) = 3;cc(1,2) = 3;
	cc(2,0) = 3;cc(2,1) = 3;cc(2,2) = 3;


	v.push_back(aa);
	v.push_back(bb);
	v.push_back(cc);

	printm(block_diag(v)); */



	test_end("ML-LART");

}
