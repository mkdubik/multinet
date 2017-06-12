#include "test.h"
#include <unordered_set>
#include <string>
#include "../include/multinet.h"
#include "../include/community/pmm.h"

using namespace mlnet;

void test_pmm() {

	test_begin("ML-PMM");

	pmm p;
	//MLNetworkSharedPtr mnet3 = read_multilayer("/home/mikki/Downloads/fftwyt.mpx","toy",',');
	MLNetworkSharedPtr mnet3 = read_multilayer("/home/guest/multinet-evaluation/data/1k_mix01","sample",' ');
	unsigned int k = 10;
	unsigned int ell = 20;
	p.fit(mnet3, k, ell);

	std::cout << "done!" << std::endl;
	test_end("ML-PMM");


}


