#include "test.h"
#include <unordered_set>
#include <string>
#include "../include/multinet.h"
#include "../include/community/glouvain.h"
#include "../include/community.h"

using namespace mlnet;


mlnet::CommunityStructureSharedPtr read_truth(mlnet::MLNetworkSharedPtr mnet) {
	std::string truth_path = "truth/";

	std::fstream myfile("/home/guest/multinet-evaluation/truth/aucs", std::ios_base::in);
	int actor;
	int community;

	mlnet::hash_map<long,std::set<mlnet::NodeSharedPtr> > result;

	while (myfile >> actor) {
		myfile >> community;

		mlnet::ActorSharedPtr a;
		if(mnet->name == "aucs" || mnet->name == "dk") {
			a = mnet->get_actors()->get_at_index(actor);
		} else {
			a = mnet->get_actor(std::to_string(actor));
		}

		for (mlnet::LayerSharedPtr l: *mnet->get_layers()) {
			mlnet::NodeSharedPtr n = mnet->get_node(a,l);
			if (n){
				result[community].insert(n);
			}
		}
	}

	mlnet::CommunityStructureSharedPtr communities = mlnet::community_structure::create();

	for (auto pair: result) {
		mlnet::CommunitySharedPtr c = mlnet::community::create();
		for (mlnet::NodeSharedPtr node: pair.second) {
			c->add_node(node);
		}
		communities->add_community(c);
	}

	return communities;
}

void test_glouvain() {

	test_begin("ML-GLOUVAIN");

	glouvain k;
	MLNetworkSharedPtr mnet = read_multilayer("/home/guest/multinet-evaluation/data/1000_overlap","aucs",' ');
	//MLNetworkSharedPtr mnet3 = read_multilayer("/home/guest/multinet-evaluation/data/1k_mix","toy",',');

	//MLNetworkSharedPtr mnet3 = read_multilayer("/home/guest/Downloads/10k_all.txt","toy",' ');
	//MLNetworkSharedPtr mnet3 = read_multilayer("/home/guest/multinet-evaluation/data/fftwyt","toy",',');
	//MLNetworkSharedPtr mnet3 = read_multilayer("/home/mikki/Downloads/friendfeed_ita.mpx","sample",',');
	double gamma = 1.0;
	double omega = 1.0;
	std::string move = "move";


	CommunityStructureSharedPtr c = k.fit(mnet, "move" ,gamma, omega, 4000);
	CommunityStructureSharedPtr truth = read_truth(mnet);

	std::cout << modularity(mnet, c, 1) << std::endl;;
	std::cout << normalized_mutual_information(c, truth, mnet->get_nodes()->size()) << std::endl;
	//(*(k.fit(mnet, move, gamma, omega))).print(std::cout);

	test_end("ML-GLOUVAIN");

}


