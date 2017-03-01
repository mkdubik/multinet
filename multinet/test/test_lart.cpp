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

	Agglomerative Clustering unit test
	Eigen::MatrixXd tmp = Eigen::MatrixXd::Zero(5, 5);
	tmp(0, 0) = 0.0;tmp(0, 1) = 17.0;tmp(0, 2) = 21.0;tmp(0, 3) = 31.0;tmp(0, 4) = 23.0;
	tmp(1, 0) = 17.0;tmp(1, 1) = 0.0;tmp(1, 2) = 30.0;tmp(1, 3) = 34.0;tmp(1, 4) = 21.0;
	tmp(2, 0) = 21.0;tmp(2, 1) = 30.0;tmp(2, 2) = 0.0;tmp(2, 3) = 28.0;tmp(2, 4) = 39.0;
	tmp(3, 0) = 31.0;tmp(3, 1) = 34.0;tmp(3, 2) = 28.0;tmp(3, 3) = 0.0;tmp(3, 4) = 43.0;
	tmp(4, 0) = 23.0;tmp(4, 1) = 21.0;tmp(4, 2) = 39.0;tmp(4, 3) = 43.0;tmp(4, 4) = 0.0;

	*/



	test_end("ML-LART");

}
