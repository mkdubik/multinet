#include "community.h"

namespace mlnet {

#define UNUSED(x) (void)(x)

hash_set<ActorSharedPtr> lart::get_ml_community(
	MLNetworkSharedPtr mnet, uint32_t t, double eps, double gamma) {

	std::vector<Eigen::MatrixXd> a = ml_network2adj_matrix(mnet);
	Eigen::MatrixXd A = supraA(a, eps);

	size_t L = a.size();
	size_t N = a[0].rows();

	Eigen::MatrixXd sA = supraA(a, 0);
	Eigen::MatrixXd D = diagA(A);
	Eigen::MatrixXd P = D * A;

	Eigen::MatrixXd Pt = matrix_power(P, t);
	Eigen::MatrixXd Dt = Dmat(Pt, D, a.size());

	std::vector<lart::cluster> clusters = AgglomerativeClustering(Dt, sA, "average");

	vector<double> mod = modMLPX(clusters, a, sA, gamma);
	auto maxmod = std::max_element(std::begin(mod), std::end(mod));
	int maxmodix = std::distance(std::begin(mod), maxmod);

	vector<int> partition = get_partition(clusters, maxmodix, L, N);

	for (size_t i = 0; i < partition.size(); i++) {
		std::cout << partition[i] << " ";
	}
	std::cout << std::endl;

	hash_set<ActorSharedPtr> actors;
	return actors;
}

vector<int> lart::get_partition(vector<lart::cluster> clusters, int maxmodix, size_t L, size_t N) {

	struct partition {
		std::vector<int> vals;
	};

	vector<partition> parts;
	partition p;
	p.vals.resize(L * N);
	std::iota (std::begin(p.vals), std::end(p.vals), 0);
	parts.push_back(p);

	for (size_t i = L * N; i < L * N + maxmodix; i++) {
		vector<int> tmp = {clusters[i].left, clusters[i].right};
		vector<int> out;

		std::set_symmetric_difference (
			parts[i-N*L].vals.begin(), parts[i-N*L].vals.end(),
			tmp.begin(), tmp.end(),
			std::back_inserter(out));

		out.push_back(i);
		partition p;
		p.vals = out;
		parts.push_back(p);
	}

	vector<partition> r;
	vector<int> val = parts[parts.size() - 1].vals;

	for (size_t i = 0; i < val.size(); i++) {
		partition pp;
		pp.vals = clusters[val[i]].orig;
		r.push_back(pp);
	}

	size_t l = r.size();
	size_t n = L * N;

	vector<int> result(n);
	for (size_t i = 0; i < l; i++) {
		for (size_t j = 0; j < r[i].vals.size(); j++) {
			result[r[i].vals[j]] = i;
		}
	}
	return result;
}

vector<double> lart::modMLPX(vector<lart::cluster> clusters, std::vector<Eigen::MatrixXd> a, Eigen::MatrixXd& sA, double gamma) {
	vector<double> r;

	size_t L = a.size();
	size_t N = a[0].rows();

	modmat(a, sA, gamma);

	r.push_back(sA.diagonal().array().sum());


	for (size_t i = N*L; i < clusters.size(); i++) {
		cluster data = clusters[i];

		vector<int> v1 = clusters[data.left].orig;
		vector<int> v2 = clusters[data.right].orig;
		double tmp = 0.0;

		for (size_t i = 0; i < v1.size(); i++) {
			for (size_t j = 0; j < v2.size(); j++) {
				tmp += sA(v1[i], v2[j]);
			}
		}

		tmp *= 2;
		r.push_back(r[r.size() - 1] + tmp);
	}
	return r;
}

void lart::modmat(std::vector<Eigen::MatrixXd> a, Eigen::MatrixXd& sA, double gamma) {
	double twoum = sA.array().sum();

	size_t L = a.size();
	size_t N = a[0].rows();

	for (size_t i = 0; i < L; i++) {
		Eigen::MatrixXd d = a[i].array().rowwise().sum();

		//Eigen::MatrixXd outer = gamma * (() / (a[i].array().sum())).array();

		Eigen::MatrixXd	product = d * d.transpose();
		double sum = a[i].array().sum();

		Eigen::MatrixXd	s1 = product.array() / sum;
		Eigen::MatrixXd	s2 = s1.array() * gamma;

		Eigen::MatrixXd s3 = sA.block(i * N, i * N, N, N);

		sA.block(i * N, i * N, N, N) = sA.block(i * N, i * N, N, N) - s2;
	}

	sA /= twoum;

}

std::vector<lart::cluster> lart::AgglomerativeClustering(Eigen::MatrixXd Dt, Eigen::MatrixXd cn, std::string Linkage) {

	UNUSED(Linkage);

	std::vector<lart::cluster> clusters(Dt.rows());
	for (int i = 0; i < Dt.rows(); i++) {
		cluster c;
		c.left = -1;
		c.right = -1;
		c.id = i;
		c.orig.push_back(i);
		clusters[i] = c;
	}

	Eigen::MatrixXd tmp(Dt);

	std::vector<int> labels (Dt.rows());
	std::iota (std::begin(labels), std::end(labels), 0);

	for (int i = 0; i < Dt.rows() - 1; i++) {
		lart::dist d = find_dist(tmp, cn);

		if (d.left < 0) {
			// No more connected components were found
			return clusters;
		}

		/* Add a new cluster into the mix with its history
			Keep the order */
		cluster c;
		c.id = Dt.rows() + i;
		if (labels[d.left] < labels[d.right]) {
			c.left = clusters[labels[d.left]].id;
			c.right = clusters[labels[d.right]].id;
		} else {
			c.left = clusters[labels[d.right]].id;
			c.right = clusters[labels[d.left]].id;
		}

		/* Merge its history */
		c.orig.insert(c.orig.begin(), clusters[labels[d.left]].orig.begin(), clusters[labels[d.left]].orig.end());
		c.orig.insert(c.orig.end(), clusters[labels[d.right]].orig.begin(), clusters[labels[d.right]].orig.end());

		clusters.push_back(c);


		/* Update the distance matrix */
		average_linkage(tmp, clusters, d);
		/* Remove the merged node from the distance matrix */
		removeEntry(tmp, d.right);
		//removeRow(tmp, d.right);
		//removeColumn(tmp, d.right);

		/* Update labels. If 0 merges with 10, we remove 10 from the pool and merge with 0 and it becomes cluster number 11 */
		labels[d.left] = Dt.rows() + i;
		labels.erase(labels.begin() + d.right);
	}


	return clusters;
}

lart::dist lart::find_dist(Eigen::MatrixXd Dt, Eigen::MatrixXd cm) {
	lart::dist d;
	d.left = -1;
	d.right = -1;
	d.val = 1000;

	for (int i = 0; i < Dt.rows(); i++) {
		for (int j = i + 1; j < Dt.cols(); j++) {
			if (Dt(i, j) < d.val && Dt(i,j) != 0 && cm(i, j) )  {
				d.left = i;
				d.right = j;
				d.val = Dt(i, j);
			}
		}
	}

	return d;
}

void lart::average_linkage(Eigen::MatrixXd& Dt, std::vector<lart::cluster> clusters, lart::dist d) {
	size_t x = clusters[d.left].orig.size();
	size_t y = clusters[d.right].orig.size();

	for (int i = 0; i < Dt.rows(); i++) {
		Dt(d.left, i) = ((Dt(d.left, i) * x) + (Dt(i, d.right) * y)) / (x + y);
		Dt(i, d.left) = Dt(d.left, i);
	}
}

void lart::removeEntry(Eigen::MatrixXd& matrix, unsigned int entryToRemove)
{
	// TODO: Is it possible to only call conservativeResize once and remove row + column?
	unsigned int numRows = matrix.rows() - 1;
	unsigned int numCols = matrix.cols();

	if(entryToRemove < numRows ){
		matrix.block(entryToRemove,0,numRows-entryToRemove,numCols) = matrix.block(entryToRemove+1,0,numRows-entryToRemove,numCols);
	}

	matrix.conservativeResize(numRows,numCols);

	numCols -= 1;
	if(entryToRemove < numCols ) {
		matrix.block(0,entryToRemove,numRows,numCols-entryToRemove) = matrix.block(0,entryToRemove+1,numRows,numCols-entryToRemove);
	}

	matrix.conservativeResize(numRows,numCols);
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

	//TODO use eigens features
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
		auto X = newP.block(i * N, 0, N, N * L);
		auto m = pairwise_distance(X, X);
		for (int j = 0; j < m.rows(); j++) {
			for (int k = 0; k < m.cols(); k++) {
				Dmat(j + (i * N), k + (i * N)) = m(j, k);
			}
		}
	}

	for (size_t i = 0; i  < L - 1; ++i) {
		for (size_t j = i + 1; j < L; ++j) {
			if (i != j) {
				auto newx = newP.block(i * N, i * N, N, N);
				auto newy = newP.block(j * N, j * N, N, N);

				auto tnewx = newP.block(j * N, i * N, N, N);
				auto tnewy = newP.block(i * N, j * N, N, N);

				Eigen::MatrixXd m1(newx.rows(), newx.cols()+tnewy.cols());
				Eigen::MatrixXd m2(newy.rows(), newy.cols()+tnewx.cols());

				m1 << newx,tnewy;
				m2 << newy,tnewx;

				auto dmat = pairwise_distance(m1, m2);
				Dmat.block(i * N, (i+1)*N, N, N) = dmat;
				Dmat.block((i+1)*N, i * N, N, N) = dmat.transpose();
			}
		}
	}
	return Dmat;
}

Eigen::MatrixXd lart::pairwise_distance(Eigen::MatrixXd X, Eigen::MatrixXd Y) {

	Eigen::MatrixXd XX = (X.array() * X.array()).rowwise().sum();
	Eigen::MatrixXd YY = (Y.array() * Y.array()).rowwise().sum().transpose();
	Eigen::MatrixXd distances = (X * Y.transpose()).unaryExpr([](const double x) { return x * -2;});

	for (int i = 0; i < distances.rows(); i++) {
		distances.col(i).array() += XX.array();
		distances.row(i).array() += YY.array();
		distances(i, i) = 0.0;
	}
	return distances.array().sqrt();
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

