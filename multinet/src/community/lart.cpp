#include "community.h"

#include <Eigen/Dense>

namespace mlnet {


hash_set<ActorSharedPtr> lart::get_ml_community(
	MLNetworkSharedPtr mnet, uint32_t t, double eps, double gamma) {



	std::vector<Eigen::MatrixXd> a = ml_network2adj_matrix(mnet);
	Eigen::MatrixXd A = supraA(a, eps);
	Eigen::MatrixXd D = diagA(A);
	Eigen::MatrixXd P = D * A;

	//prcheck(a, P);
	Eigen::MatrixXd Pt = matrix_power(P, t);
	Eigen::MatrixXd Dt = Dmat(Pt, D, a.size());

	hash_set<ActorSharedPtr> actors;
	return actors;
}

std::vector<Eigen::MatrixXd> lart::ml_network2adj_matrix(MLNetworkSharedPtr mnet) {

	size_t N = mnet->get_layers()->size();
	size_t M = mnet->get_actors()->size();

	std::vector<Eigen::MatrixXd> adj(N);

	for (LayerSharedPtr l: *mnet->get_layers()) {
		Eigen::MatrixXd m = Eigen::MatrixXd::Zero(M, M);

		for (EdgeSharedPtr e: *mnet->get_edges(l, l)) {
			int v1_id = e->v1->actor->id;
			int v2_id = e->v2->actor->id;
			m(v1_id - 1, v2_id - 1) = 1;
			m(v2_id - 1, v1_id - 1) = 1;
		}
		adj[l->id - 1] = m;
	}

	return adj;
}

Eigen::MatrixXd lart::block_diag(std::vector<Eigen::MatrixXd> a) {

	Eigen::MatrixXd out = Eigen::MatrixXd::Zero(a[0].rows() * a.size(), a[0].cols() * a.size());

	size_t r, c;
	r = 0;
	c = 0;


	for (size_t i = 0; i < a.size(); i++) {
		out.block(r, c, a[i].rows(), a[i].cols()) = a[i];
		r += a[i].rows();
		c += a[i].cols();
	}
	return out;
}


Eigen::MatrixXd lart::supraA(std::vector<Eigen::MatrixXd> a, double eps) {
	Eigen::MatrixXd A = block_diag(a);

	for (int i = 0; i < A.rows(); ++i) {
		A(i, i) = eps;
	}

	size_t L = a.size();
	size_t N = a[0].rows();

	for (size_t i = 0; i  < L - 1; ++i) {
		for (size_t j = i + 1; j < L; ++j) {
			Eigen::MatrixXd d = (a[i].array() * a[j].array()).colwise().sum();

			auto tmp1 = A.block(i * N, (i + 1) * N, a[i].rows(), a[i].cols());
			for (int k = 0; k < tmp1.rows(); k++) {
				tmp1(k, k) = d(k) + eps;
			}

			auto tmp2 = A.block((i + 1) * N, i * N, a[j].rows(), a[j].cols());
			for (int k = 0; k < tmp2.rows(); k++) {
				tmp2(k, k) = d(k) + eps;
			}
		}
	}

	return A;
}


Eigen::MatrixXd lart::diagA(Eigen::MatrixXd a) {

	Eigen::MatrixXd A = Eigen::MatrixXd::Zero(a.rows(), a.cols());
	Eigen::MatrixXd d = a.colwise().sum();

	for (int i = 0; i < A.cols(); i++) {
		if (d(i) > 0) {
			A(i, i) = 1 / d(i);
		} else {
			A(i, i) = 0;
		}
	}
	return A;
}

Eigen::MatrixXd lart::Dmat(Eigen::MatrixXd Pt, Eigen::MatrixXd D, size_t L) {

	size_t N = Pt.rows() / L;

	Eigen::MatrixXd D_sqrt = D.array().sqrt();
	Eigen::MatrixXd newP = Pt * D_sqrt;
	Eigen::MatrixXd Dmat = 	Eigen::MatrixXd::Zero(N * L, N * L);

	for (size_t i = 0; i  < L - 1; ++i) {
		auto k = pairwise_distance(newP.block(0, 0, 16, 32));
		std::cout << k.rows() << k.cols() << std::endl;
	}

	for (size_t i = 0; i  < L - 1; ++i) {
		for (size_t j = i + 1; j < L; ++j) {

		}
	}
	return Dmat;
}

Eigen::MatrixXd lart::pairwise_distance(Eigen::MatrixXd X) {

	Eigen::MatrixXd XX = (X.array() * X.array()).rowwise().sum();
	Eigen::MatrixXd YY = XX.transpose();
	Eigen::MatrixXd distances = (X * X.transpose()).unaryExpr([](const double x) { return x * -2;});

	Eigen::IOFormat f(Eigen::StreamPrecision, 0, ",", ",", "[", "]", "[", "]");
	for (int i = 0; i < distances.rows(); i++) {
		distances.col(i).array() += XX.array();
		distances.row(i).array() += YY.array();
		//distances(i, i) = 0.0;
	}
	return distances.array().sqrt();
}


void lart::prcheck(std::vector<Eigen::MatrixXd> a, Eigen::MatrixXd P) {

	//ublas::matrix<double> g = sum(a, 0);
	//TODO
}

Eigen::MatrixXd lart::matrix_power(Eigen::MatrixXd m, uint32_t t) {

	if (t == 0) {
		return Eigen::MatrixXd::Identity(m.rows(), m.cols());
	}

	Eigen::MatrixXd Dt(m);

	for (uint32_t i = 1; i < t; i++) {
		Dt = Dt * m;
	}
	return Dt;
}

}

