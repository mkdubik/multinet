#include "community.h"
#include <unsupported/Eigen/MatrixFunctions>

namespace mlnet {

#define UNUSED(x) (void)(x)

hash_set<ActorSharedPtr> lart::get_ml_community(
	MLNetworkSharedPtr mnet, uint32_t t, double eps, double gamma) {

	UNUSED(t);UNUSED(eps);UNUSED(gamma);

	hash_set<ActorSharedPtr> actors;

	std::vector<Eigen::SparseMatrix<double>> a = ml_network2adj_matrix(mnet);

	if (a.size() < 1) {
		return actors;
	}

	//Eigen::IOFormat f(Eigen::StreamPrecision, 0, ", ", ",", "[", "]", "[", "]");

	std::cout << " A" << std::endl;
	Eigen::SparseMatrix<double> sA = supraA(a, eps);
	std::cout << " B" << std::endl;
	Eigen::SparseMatrix<double> sA0 = supraA(a, 0);
	std::cout << " C" << std::endl;
	Eigen::SparseMatrix<double> dA = diagA(sA);
	std::cout << " D" << std::endl;
	Eigen::SparseMatrix<double> aP = dA * sA;
	std::cout << " E" << std::endl;
	Eigen::SparseMatrix<double> Pt = matrix_power(aP, t);
	std::cout << " F" << std::endl;
	Eigen::SparseMatrix<double> Dt = Dmat(Pt, dA, a.size());

	/*
	Eigen::MatrixXd P = D * A;

	Eigen::MatrixXd Pt = matrix_power(P, t);

	std::vector<lart::cluster> clusters = AgglomerativeClustering(Dt, sA, "average");

	vector<double> mod = modMLPX(clusters, a, sA, gamma);
	auto maxmod = std::max_element(std::begin(mod), std::end(mod));
	int maxmodix = std::distance(std::begin(mod), maxmod);

	vector<int> partition = get_partition(clusters, maxmodix, L, N);

	for (size_t i = 0; i < partition.size(); i++) {
		std::cout << partition[i] << " ";
	}
	std::cout << std::endl;*/

	return actors;
}

std::vector<Eigen::SparseMatrix<double>> lart::ml_network2adj_matrix(MLNetworkSharedPtr mnet) {

	DTRACE0(ML2AM_START);

	size_t N = mnet->get_layers()->size();
	size_t M = mnet->get_actors()->size();

	std::vector<Eigen::SparseMatrix<double>> adj(N);

	for (LayerSharedPtr l: *mnet->get_layers()) {
		Eigen::SparseMatrix<double> m = Eigen::SparseMatrix<double> (M, M);
		m.reserve(Eigen::VectorXi::Constant(M / 2, M / 2));

		for (EdgeSharedPtr e: *mnet->get_edges(l, l)) {
			int v1_id = e->v1->actor->id;
			int v2_id = e->v2->actor->id;
			m.insert(v1_id - 1, v2_id - 1) = 1;
			m.insert(v2_id - 1, v1_id - 1) = 1;
		}
		adj[l->id - 1] = m;
	}

	std::cout << " Done reading" << std::endl;

	DTRACE0(ML2AM_END);
	return adj;
}

Eigen::SparseMatrix<double> lart::block_diag(std::vector<Eigen::SparseMatrix<double>> a) {

	Eigen::SparseMatrix<double> m = Eigen::SparseMatrix<double>(
		a[0].rows() * a.size(), a[0].cols() * a.size());

	m.reserve(Eigen::VectorXi::Constant(a[0].rows(), a[0].rows()));

	size_t r, c;
	r = 0;
	c = 0;

	std::vector<Eigen::Triplet<double>> tlist;
	tlist.reserve(a[0].rows() * a.size());

	for (size_t i = 0; i < a.size(); i++) {
		for (int j = 0; j < a[i].outerSize(); j++) {
			for (Eigen::SparseMatrix<double>::InnerIterator it(a[i], j); it; ++it) {
				tlist.push_back(Eigen::Triplet<double>(r + it.row(), c + it.col(), it.value()));
			}
		}
		r += a[i].rows();
		c += a[i].cols();
	}
	m.setFromTriplets(tlist.begin(), tlist.end());
	return m;
}

Eigen::SparseMatrix<double> lart::supraA(std::vector<Eigen::SparseMatrix<double>> a, double eps) {
	Eigen::SparseMatrix<double> A = block_diag(a);

	size_t L = a.size();
	size_t N = a[0].rows();

	for (size_t i = 0; i  < L - 1; ++i) {
		for (size_t j = i + 1; j < L; ++j) {
			Eigen::RowVectorXd d = sum(a[i].cwiseProduct(a[j]), 1);

			std::vector<Eigen::Triplet<double>> tlist;
			tlist.reserve(a[i].rows());

			for (int j = 0; j < A.outerSize(); j++) {
				for (Eigen::SparseMatrix<double>::InnerIterator it(A, j); it; ++it) {
					tlist.push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value()));
				}
			}

			int ix_a = i * N;
			int ix_b = (i + 1) * N;

			for (int k = 0; k < a[i].rows(); k++) {
				double intra = d(k) + eps;
				tlist.push_back(Eigen::Triplet<double>(ix_a + k, ix_b + k, intra));
				tlist.push_back(Eigen::Triplet<double>(ix_b + k, ix_a + k, intra));
			}

			for (int k = 0; k < A.rows(); k++) {
				tlist.push_back(Eigen::Triplet<double>(k, k, eps));
			}
			A.setFromTriplets(tlist.begin(), tlist.end());
		}
	}
	return A;
}

Eigen::SparseMatrix<double> lart::diagA(Eigen::SparseMatrix<double> A) {
	Eigen::SparseMatrix<double> dA = Eigen::SparseMatrix<double>(A.rows(), A.cols());
	dA.reserve(Eigen::VectorXi::Constant(A.rows() / 2, A.rows() / 2));

	Eigen::RowVectorXd d = sum(A, 1);

	std::vector<Eigen::Triplet<double>> tlist;
	tlist.reserve(A.rows());

	for (int k = 0; k < A.rows(); k++) {
		tlist.push_back(Eigen::Triplet<double>(k, k, 1 / d[k]));
	}
	dA.setFromTriplets(tlist.begin(), tlist.end());
	return dA;
}

Eigen::SparseMatrix<double> lart::Dmat(Eigen::SparseMatrix<double> Pt, Eigen::SparseMatrix<double> dA, size_t L) {
	// NOTE: dA side effect

	std::cout << " Getting SQRT of dA..." << std::endl;

	for (int j = 0; j < dA.outerSize(); j++) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(dA, j); it; ++it) {
			dA.coeffRef(it.row(), it.col()) = std::sqrt(it.value());
		}
	}

	std::cout << " Done with SQRT" << std::endl;

	Eigen::SparseMatrix<double> newP = Pt * dA;

	Eigen::SparseMatrix<double> Dmat = Eigen::SparseMatrix<double>(dA.rows(), dA.cols());
	Dmat.reserve(Eigen::VectorXi::Constant(dA.rows() / 2, dA.rows() / 2));
	size_t N = Pt.rows() / L;

	for (size_t i = 0; i < L; ++i) {
		auto X = newP.block(i * N, 0, N, N * L);
		auto m = pairwise_distance(X, X);

		/*for (int j = 0; j < m.rows(); j++) {
			for (int k = 0; k < m.cols(); k++) {
				Dmat(j + (i * N), k + (i * N)) = m(j, k);
			}
		}*/
	}

	/*
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
	}*/
	return Dmat;
}

Eigen::SparseMatrix<double> lart::pairwise_distance(Eigen::SparseMatrix<double> X, Eigen::SparseMatrix<double> Y) {

	Eigen::RowVectorXd XX = sum(X.cwiseProduct(X), 0);
	Eigen::RowVectorXd YY = sum(Y.cwiseProduct(Y), 0).transpose();

	Eigen::SparseMatrix<double> distances = (X * Y.transpose()).unaryExpr([](const double x) { return x * -2;});

	for (int i = 0; i < distances.rows(); i++) {
		//distances.col(i) += YY;

		//for (int j = 0; j < distances.cols(); i++) {
			//dA.coeffRef(it.row(), it.col()) = std::sqrt(it.value());
		//}
	}
		/*
		std::vector<Eigen::Triplet<double>> tlistCol; tlistCol.reserve(X.rows());
		for (int j = 0; j < distances.cols(); j++) {
			tlistCol.push_back(Eigen::Triplet<double>(j, i, distances.coeffRef(j, i) + XX.coeffRef(j, 0)));
		}
		distances.setFromTriplets(tlistCol.begin(), tlistCol.end());

		std::vector<Eigen::Triplet<double>> tlistRow; tlistRow.reserve(X.rows());
		for (int j = 0; j < distances.cols(); j++) {
			tlistRow.push_back(Eigen::Triplet<double>(i, j, distances.coeffRef(i, j) + YY.coeffRef(0, j)));
		}
		distances.setFromTriplets(tlistRow.begin(), tlistRow.end());

		//distances.col(i).array() += XX.array();
		//distances.row(i).array() += YY.array();
		//distances.insert(i, i) = 0.0;
	}


	std::cout << std::endl;

	std::vector<double> tmp1(X.rows(), 0);
	for (int j = 0; j < tmp.outerSize(); j++) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(tmp, j); it; ++it) {
			tmp1[it.row()] = it.value() + tmp1[it.row()];
		}
	}

	std::vector<Eigen::Triplet<double>> tlist;
	tlist.reserve(X.rows());
	for (unsigned int j = 0; j < tmp1.size(); j++) {
		tlist.push_back(Eigen::Triplet<double>(j, 1, tmp1[j]));
	}
	XX.setFromTriplets(tlist.begin(), tlist.end());*/


	/*Eigen::SparseMatrix<double> YY = pairwise_distance_sum(Y).transpose();




	return distances.array().sqrt();*/
}


Eigen::RowVectorXd lart::sum(Eigen::SparseMatrix<double> X, int axis) {

	Eigen::RowVectorXd d (X.rows());

	for (int i = 0; i < X.outerSize(); i++) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(X, i); it; ++it) {
			if (axis){
				d(it.col()) += it.value();
			} else {
				d(it.row()) += it.value();
			}
		}
	}
	return d;
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



Eigen::SparseMatrix<double> lart::matrix_power(Eigen::SparseMatrix<double> m, uint32_t t) {

	if (t == 0) {
		Eigen::SparseMatrix<double> A(m.rows(), m.cols());
		A.setIdentity();
		return A;
	}

	Eigen::SparseMatrix<double> Dt(m);
	for (uint32_t i = 1; i < t; i++) {
		// TODO Write more and consider the pruning
		Dt = (Dt * m).pruned(0.001, 10);
		Dt.makeCompressed();
	}
	return Dt;
}

}

