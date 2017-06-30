#include "test.h"
#include <unordered_set>
#include <string>
#include "../include/community/lart.h"
#include "../include/community/abacus.h"

using namespace mlnet;

mlnet::CommunityStructureSharedPtr read_truth_overlap(mlnet::MLNetworkSharedPtr mnet) {
	std::string truth_path = "truth/";

	std::fstream file("/home/guest/multinet-evaluation/truth/1000_overlap", std::ios_base::in);
	std::map<unsigned int, std::vector<unsigned int>> mapped;

	std::string str;
	while(getline(file, str))
	{
		std::istringstream ss(str);
		unsigned int num;

		int actor_id = 0;
		/*
		map from:
		aid cid cid cid
		to:
		cid aid aid aid */

		while(ss >> num) {
			if (actor_id == 0){
				actor_id = num;
			} else {
				mapped[num].push_back(actor_id);
			}
		}

	}

	mlnet::CommunityStructureSharedPtr communities = mlnet::community_structure::create();
	for(const auto &p : mapped) {
		mlnet::CommunitySharedPtr c = mlnet::community::create();

		for (auto m: p.second) {
			for (mlnet::NodeSharedPtr n : *mnet->get_nodes(((
				mnet->get_actor(std::to_string(m)))))) {
				(*c).add_node(n);
			}
		}
		(*communities).add_community(c);
	}

	return communities;
}


void test_lart() {

	test_begin("ML-LART");


	lart k;

	MLNetworkSharedPtr mnet = read_multilayer("/home/guest/multinet-evaluation/data/1000_overlap","aucs",' ');
	//MLNetworkSharedPtr mnet3 = read_multilayer("/home/guest/multinet-evaluation/data/1k_mix","toy",',');

	//MLNetworkSharedPtr mnet3 = read_multilayer("/home/guest/Downloads/10k_all.txt","toy",' ');
	//MLNetworkSharedPtr mnet3 = read_multilayer("/home/guest/multinet-evaluation/data/fftwyt","toy",',');
	//MLNetworkSharedPtr mnet3 = read_multilayer("/home/mikki/Downloads/friendfeed_ita.mpx","sample",',');
	uint32_t t = 9;
	double eps = 1;
	double gamma = 1;



	CommunityStructureSharedPtr c = k.fit(mnet, t, eps, gamma);
	CommunityStructureSharedPtr truth = read_truth_overlap(mnet);

	std::cout << modularity(mnet, c, 1) << std::endl;
	std::cout << modularity(mnet, truth, 1) << std::endl;;

	std::cout << normalized_mutual_information(c, truth, mnet->get_nodes()->size()) << std::endl;

	//std::ofstream out("/home/guest/multinet/multinet/test/DK_Pol_lart.txt");
	//(*c).print(std::cout);

	test_end("ML-LART");

}


