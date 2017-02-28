#include "community.h"
#include <iomanip>
#include <Eigen/Dense>
#include <map>

namespace mlnet {

#define UNUSED(x) (void)(x)

//double (lart::*distance_fptr)(Eigen::MatrixXd Dt);

hash_set<ActorSharedPtr> lart::get_ml_community(
	MLNetworkSharedPtr mnet, uint32_t t, double eps, double gamma) {


	std::setprecision(11);
	std::cout.precision(11);

	std::vector<Eigen::MatrixXd> a = ml_network2adj_matrix(mnet);
	Eigen::MatrixXd A = supraA(a, eps);
	Eigen::MatrixXd A0 = supraA(a, 0);

	Eigen::MatrixXd D = diagA(A);
	Eigen::MatrixXd P = D * A;

	//prcheck(a, P);
	Eigen::MatrixXd Pt = matrix_power(P, t);
	Eigen::MatrixXd Dt = Dmat(Pt, D, a.size());

	UNUSED(Dt);
	UNUSED(gamma);
	UNUSED(A0);


	Eigen::IOFormat f(Eigen::StreamPrecision, 0, ",", ",", "[", "]", "[", "]");

	//updateDt(Dt, A0);

	Eigen::MatrixXd tmp = Eigen::MatrixXd::Zero(5, 5);

	tmp(0, 0) = 0.0;tmp(0, 1) = 17.0;tmp(0, 2) = 21.0;tmp(0, 3) = 31.0;tmp(0, 4) = 23.0;
	tmp(1, 0) = 17.0;tmp(1, 1) = 0.0;tmp(1, 2) = 30.0;tmp(1, 3) = 34.0;tmp(1, 4) = 21.0;
	tmp(2, 0) = 21.0;tmp(2, 1) = 30.0;tmp(2, 2) = 0.0;tmp(2, 3) = 28.0;tmp(2, 4) = 39.0;
	tmp(3, 0) = 31.0;tmp(3, 1) = 34.0;tmp(3, 2) = 28.0;tmp(3, 3) = 0.0;tmp(3, 4) = 43.0;
	tmp(4, 0) = 23.0;tmp(4, 1) = 21.0;tmp(4, 2) = 39.0;tmp(4, 3) = 43.0;tmp(4, 4) = 0.0;

	AgglomerativeClustering(Dt, "average");

	hash_set<ActorSharedPtr> actors;
	return actors;
}

void lart::updateDt(Eigen::MatrixXd Dt, Eigen::MatrixXd A0) {
	UNUSED(Dt);
	UNUSED(A0);

}


lart::pair lart::find_smallest_ix(Eigen::MatrixXd Dt) {

	double smallest = 100;
	int ix_x = 0;
	int ix_y = 0;

	for (int i = 0; i < Dt.rows(); i++) {
		for (int j = i + 1; j < Dt.cols(); j++) {
			if (Dt(i, j) < smallest) {
				smallest = Dt(i, j);
				ix_x = i;
				ix_y = j;
			}
		}
	}

	pair result;
	result.ix_x = ix_x;
	result.ix_y = ix_y;
	result.smallest = smallest;
	return result;
}

void lart::removeRow(Eigen::MatrixXd& matrix, unsigned int rowToRemove)
{
	unsigned int numRows = matrix.rows()-1;
	unsigned int numCols = matrix.cols();

	if( rowToRemove < numRows )
	    matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.block(rowToRemove+1,0,numRows-rowToRemove,numCols);

	matrix.conservativeResize(numRows,numCols);
}

void lart::removeColumn(Eigen::MatrixXd& matrix, unsigned int colToRemove)
{
	unsigned int numRows = matrix.rows();
	unsigned int numCols = matrix.cols()-1;

	if( colToRemove < numCols )
	    matrix.block(0,colToRemove,numRows,numCols-colToRemove) = matrix.block(0,colToRemove+1,numRows,numCols-colToRemove);

	matrix.conservativeResize(numRows,numCols);
}

void lart::updateDt(Eigen::MatrixXd& Dt, lart::pair p, std::vector<int> merges) {
	for (int i = 0; i < Dt.rows(); i++) {
		if (i == p.ix_x || i == p.ix_y) {
			continue;
		}

		//int x = merges[p.ix_x];
		//int y = merges[p.ix_y];


		Dt(p.ix_x, i) = ((Dt(p.ix_x, i)) + (Dt(i, p.ix_y))) / 2.0;
		Dt(i, p.ix_x) = Dt(p.ix_x, i);

	}

}

std::vector<std::vector<int>> lart::AgglomerativeClustering(Eigen::MatrixXd Dt, std::string Linkage) {

	std::vector<std::vector<int>> ids;

	std::vector<int> labels (Dt.rows());
	std::iota (std::begin(labels), std::end(labels), 0);
	std::vector<int> merges (Dt.rows(), 1);

	Eigen::IOFormat f(Eigen::StreamPrecision, 0, ",", ",", "[", "]", "[", "]");
	Eigen::MatrixXd tmp(Dt);


	for (int i = 0; i < Dt.rows() - 1; i++) {
		pair result = find_smallest_ix(tmp);

		std::vector<int> v = {labels[result.ix_x], labels[result.ix_y]};
		std::sort(v.begin(), v.end());
		ids.push_back(v);

		labels[result.ix_x] = Dt.rows() + i;
		labels.erase(labels.begin() + result.ix_y);


		updateDt(tmp, result, merges);

		//merges[result.ix_x]++;
		//merges[result.ix_x] = merges[result.ix_x] + merges[result.ix_y];
		//merges.erase(merges.begin() + result.ix_y);


		removeRow(tmp, result.ix_y);
		removeColumn(tmp, result.ix_y);

	}

	for (int i = 0; i < ids.size(); i++) {
		for (int j = 0; j < ids[i].size(); j++) {
			std::cout << ids[i][j] << ",";
		}
		std::cout << std::endl;
	}


	return ids;
	//map<std::string, >

}


std::vector<Eigen::MatrixXd> lart::ml_network2adj_matrix(MLNetworkSharedPtr mnet) {

	DTRACE0(ML2AM_START);
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

	DTRACE0(ML2AM_END);
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

	for (size_t i = 0; i < L; ++i) {
		// TODO Fix the magic number
		auto m = pairwise_distance(newP.block(i * N, 0, 16, 32));
		for (int j = 0; j < m.rows(); j++) {
			for (int k = 0; k < m.cols(); k++) {
				Dmat(j + (i * N), k + (i * N)) = m(j, k);
			}
		}
	}

	for (size_t i = 0; i  < L - 1; ++i) {
		for (size_t j = i + 1; j < L; ++j) {
			if (i != j) {
				auto newx = newP.block(i * N, i * N, (i+1)*N, (i+1)*N);
				auto newy = newP.block(j * N, j * N, (i+1)*N, (i+1)*N);

				auto tnewx = newP.block(j * N, i * N, (i+1)*N, (i+1)*N);
				auto tnewy = newP.block(i * N, j * N, (i+1)*N, (i+1)*N);

				Eigen::MatrixXd m1(newx.rows(), newx.cols()+tnewy.cols());
				Eigen::MatrixXd m2(newy.rows(), newy.cols()+tnewx.cols());

				m1 << newx,tnewy;
				m2 << newy,tnewx;

				//TODO missing if len(l) > 0

				auto dmat = pairwise_distance(m1, m2);
				Dmat.block(i * N, (i+1)*N, (i+1)*N, (i+1)*N) = dmat;
				Dmat.block((i+1)*N, i * N, (i+1)*N, (i+1)*N) = dmat.transpose();
			}
		}
	}
	return Dmat;
}

Eigen::MatrixXd lart::pairwise_distance(Eigen::MatrixXd X) {

	Eigen::MatrixXd XX = (X.array() * X.array()).rowwise().sum();
	Eigen::MatrixXd YY = XX.transpose();
	Eigen::MatrixXd distances = (X * X.transpose()).unaryExpr([](const double x) { return x * -2;});

	for (int i = 0; i < distances.rows(); i++) {
		distances.col(i).array() += XX.array();
		distances.row(i).array() += YY.array();
		//distances(i, i) = 0.0;
	}
	return distances.array().sqrt();
}

Eigen::MatrixXd lart::pairwise_distance(Eigen::MatrixXd X, Eigen::MatrixXd Y) {

	Eigen::MatrixXd XX = (X.array() * X.array()).rowwise().sum();
	Eigen::MatrixXd YY = (Y.array() * Y.array()).rowwise().sum().transpose();
	Eigen::MatrixXd distances = (X * Y.transpose()).unaryExpr([](const double x) { return x * -2;});

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
	UNUSED(a);
	UNUSED(P);

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

