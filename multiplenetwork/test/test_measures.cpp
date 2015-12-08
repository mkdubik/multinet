/*
 * testNetwork.cpp
 *
 * Created on: Feb 7, 2014
 * Author: matteomagnani
 * Version: 0.0.1
 */

#include "test.h"
#include "mlnetwork.h"
#include <iostream>
#include <string>

using namespace mlnet;

void test_measures() {


	test_begin("Actor measures");
	std::cout << "Reading the multilayer network...";
	MLNetworkSharedPtr mnet = read_multilayer("test/io2.mpx","mlnet 2",',');
	std::cout << "done!" << std::endl;
	LayerSharedPtr l1 = mnet->get_layer("l1");
	LayerSharedPtr l2 = mnet->get_layer("l2");
	ActorSharedPtr u1 = mnet->get_actor("U1");
	ActorSharedPtr u3 = mnet->get_actor("U3");
	std::unordered_set<LayerSharedPtr> layers = {l1,l2};

	std::cout << "Testing degree...";
	if (degree(mnet,u1,l1,INOUT) != 3) throw FailedUnitTestException("Wrong degree, actor U1, layer l1: " + to_string(degree(mnet,u1,l1,INOUT)));
	if (degree(mnet,u1,layers,INOUT) != 5) throw FailedUnitTestException("Wrong degree, both layers: " + to_string(degree(mnet,u1,layers,INOUT)));
	std::cout << "done!" << std::endl;

	std::cout << "Testing degree mean and degree deviation...";
	if (degree_mean(mnet,u1,layers,INOUT) != 2.5) throw FailedUnitTestException("Wrong mean degree, actor U1: " + to_string(degree_mean(mnet,u1,layers,INOUT)));
	if (degree_deviation(mnet,u1,layers,INOUT) != .5) throw FailedUnitTestException("Wrong degree deviation, actor U1: " + to_string(degree_deviation(mnet,u1,layers,INOUT)));
	std::cout << "done!" << std::endl;

	std::cout << "Testing neighborhood...";
	if (neighbors(mnet,u1,l1,INOUT).size() != 3) throw FailedUnitTestException("Wrong neighborhood, layer l1: " + to_string(neighbors(mnet,u1,l1,INOUT).size()));
	if (neighbors(mnet,u3,l1,IN).size() != 1) throw FailedUnitTestException("Wrong in_neighborhood, layer l1: " + to_string(neighbors(mnet,u3,l1,IN).size()));
	if (neighbors(mnet,u3,l1,OUT).size() != 1) throw FailedUnitTestException("Wrong out_neighborhood, layer l1: " + to_string(neighbors(mnet,u3,l1,OUT).size()));
	if (neighbors(mnet,u3,l2,IN).size() != 1) throw FailedUnitTestException("Wrong in_neighborhood, layer l2: " + to_string(neighbors(mnet,u3,l2,IN).size()));
	if (neighbors(mnet,u3,l2,OUT).size() != 1) throw FailedUnitTestException("Wrong out_neighborhood, layer l2: " + to_string(neighbors(mnet,u3,l2,OUT).size()));
	if (neighbors(mnet,u1,layers,INOUT).size() != 4) throw FailedUnitTestException("Wrong neighborhood, both layers: " + to_string(neighbors(mnet,u1,layers,INOUT).size()));
	std::cout << "done!" << std::endl;

	std::cout << "Testing exclusive neighborhood...";
	if (xneighbors(mnet,u1,l1,INOUT).size() != 2) throw FailedUnitTestException("Wrong exclusive neighborhood, layer l1: " + to_string(xneighbors(mnet,u1,l1,INOUT).size()));
	if (xneighbors(mnet,u1,layers,INOUT).size() != 4) throw FailedUnitTestException("Wrong exclusive neighborhood, both layers: " + to_string(xneighbors(mnet,u1,layers,INOUT).size()));
	std::cout << "done!" << std::endl;

	std::cout << "Testing layer relevance...";
	if (relevance(mnet,u1,l1,INOUT) != 3.0/4.0) throw FailedUnitTestException("Wrong relevance, layer l1: " + to_string(relevance(mnet,u1,l1,INOUT)));
	if (relevance(mnet,u1,layers,INOUT) != 1) throw FailedUnitTestException("Wrong relevance, both layers: " + to_string(relevance(mnet,u1,layers,INOUT)));
	std::cout << "done!" << std::endl;

	std::cout << "Testing exclusive layer relevance...";
	if (xrelevance(mnet,u1,l1,INOUT) != 2.0/4.0) throw FailedUnitTestException("Wrong exclusive relevance, layer l1: " + to_string(xrelevance(mnet,u1,l1,INOUT)));
	if (xrelevance(mnet,u1,layers,INOUT) != 1) throw FailedUnitTestException("Wrong exclusive relevance, both layers: " + to_string(xrelevance(mnet,u1,layers,INOUT)));
	std::cout << "done!" << std::endl;

	std::cout << "Testing jaccard similarity...";
	MLNetworkSharedPtr mnet1 = read_multilayer("test/io1.mpx","mlnet 1",',');
	if (jaccard_similarity(mnet,layers) != 1.0/10.0) throw FailedUnitTestException("Wrong layer similarity");
	std::cout << "done!" << std::endl;

	std::cout << "Computing multilayer distance between all pairs of vertexes...";
	// The result is stored in the variable paths, where for each target vertex (from source U0) we obtain a set of shortest paths
	hash<ActorSharedPtr,std::set<distance> > dists = pareto_distance(mnet, mnet->get_actor("U0"));
	std::cout << "done!" << std::endl;

	for (auto p: dists) {
		for (auto dist: p.second) {
			std::cout << "- dist from U0 to " <<  p.first->name << ": " << dist.to_string() << std::endl;
		}
	}

	std::cout << "Testing sample values (U0->U3: 2 shortest paths expected)...";
	if (dists[mnet->get_actor("U3")].size()!=2) throw FailedUnitTestException("Expected 2 distances, found " + to_string(dists[mnet->get_actor("U3")].size()));
	distance p1 = *dists[mnet->get_actor("U3")].begin();
	distance p2 = *++dists[mnet->get_actor("U3")].begin();
	if (p1.length()!=2) throw FailedUnitTestException("Wrong length: distance 1, " + to_string(p1.length()));
	if (p2.length()!=2) throw FailedUnitTestException("Wrong length: distance 2, " + to_string(p2.length()));
	if (p1.length(l1,l1)!=2) throw FailedUnitTestException("Wrong number of edges: distance 1 on layer l1");
	if (p1.length(l2,l2)!=0) throw FailedUnitTestException("Wrong number of edges: distance 1 on layer l2");
	if (p2.length(l1,l1)!=1) throw FailedUnitTestException("Wrong number of edges: distance 2 on layer l1");
	if (p2.length(l2,l2)!=1) throw FailedUnitTestException("Wrong number of edges: distance 2 on layer l2");
	std::cout << "done!" << std::endl;

	test_end("Actor measures");

}

