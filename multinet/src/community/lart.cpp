#include "community.h"
#include <boost/numeric/ublas/matrix_proxy.hpp>

namespace mlnet {


hash_set<ActorSharedPtr> lart::get_ml_community(
	MLNetworkSharedPtr mnet, uint32_t t, float eps, float gamma) {

	hash_set<ActorSharedPtr> actors;

	std::vector<ublas::matrix<float>> a = ml_network2adj_matrix(mnet);
	ublas::matrix<float> A = supraA(a, eps);
	ublas::matrix<float> D = diagA(A);
	ublas::matrix<float> P = ublas::element_prod(D, A);
	//prcheck(a, P);
	ublas::matrix<float> Dt = matrix_power(P, t);


	return actors;
}

ublas::matrix<float> lart::matrix_power(ublas::matrix<float> m, uint32_t t) {

	ublas::matrix<float> Dt(m);
	for (uint32_t i = 0; i < t - 1; i++) {
		Dt = ublas::prod(Dt, m);
	}
	return m;
}

std::vector<ublas::matrix<float>> lart::ml_network2adj_matrix(MLNetworkSharedPtr mnet) {

	size_t N = mnet->get_layers()->size();
	size_t M = mnet->get_actors()->size();

	std::vector<ublas::matrix<float>> adj(N);

	for (LayerSharedPtr l: *mnet->get_layers()) {
		ublas::matrix<float> m = ublas::zero_matrix<float>(M);

		for (EdgeSharedPtr e: *mnet->get_edges(l, l)) {
			size_t v1_id = e->v1->actor->id;
			size_t v2_id = e->v2->actor->id;
			m(v1_id - 1, v2_id- 1) = 1;
			m(v2_id - 1, v1_id- 1) = 1;
		}
			adj[l->id - 1] = m;
	}
	return adj;
}

ublas::matrix<float> lart::Dmat(
	ublas::matrix<float> Dt,
	ublas::matrix<float> D)
}


ublas::matrix<float> lart::supraA(std::vector<ublas::matrix<float>> a, float eps) {

	ublas::matrix<float> A = block_diag(a);

	// Fill diagonal
	for (size_t i = 0; i < A.size1(); ++i) {
		A(i, i) = eps;
	}

	size_t L = a.size();
	size_t N = a[0].size1();

	for (size_t i = 0; i  < L - 1; ++i) {
		for (size_t j = i + 1; j < L; ++j) {
			ublas::vector<float> d = sum(ublas::element_prod(a[i], a[j]), 0);

			ublas::matrix_vector_range<ublas::matrix<float> > tmp1(A,
				ublas::range(i * N, (i + 1) * N),
				ublas::range(j * N, (j + 1) * N));

			for (size_t k = 0; k < d.size(); ++k) {
				tmp1(k) = d[k] + eps;
			}

			ublas::matrix_vector_range<ublas::matrix<float> > tmp2(A,
				ublas::range(j * N, (j + 1) * N),
				ublas::range(i * N, (i + 1) * N));

			for (size_t k = 0; k < d.size(); ++k) {
				tmp2(k) = d[k] + eps;
			}
		}
	}

	return A;
}


ublas::matrix<float> lart::diagA(ublas::matrix<float> a) {
	ublas::vector<float> d = sum(a, 0);

	for (size_t i = 0; i < d.size(); i++) {
		if (d[i] != 0) {
			a(i, i) = 1 / d[i];
		} else {
			a(i, i) = 0;
		}
	}

	return a;
}


void lart::prcheck(std::vector<ublas::matrix<float>> a,
			ublas::matrix<float> P) {

	ublas::matrix<float> g = sum(a, 0);
	//TODO
}


ublas::matrix<float> lart::sum(std::vector<ublas::matrix<float>> a, int axis) {

	ublas::matrix<float> m = ublas::zero_matrix<float>(a[0].size1(), a[0].size2());

	for (size_t i = 0; i < a.size(); i++) {
		for (size_t j = 0; j < a[i].size1(); j++) {
			for (size_t k = 0; k < a[i].size2(); k++) {
				if (axis == 0) {
					m(j, k) += a[i](j, k);
				} else {
					m(j, k) += a[i](j, k);
				}
			}
		}
	}

	return m;
}

ublas::vector<float> lart::sum(ublas::matrix<float> a, int axis) {

	ublas::vector<float> r = ublas::zero_vector<float>(a.size1());

	for (size_t i = 0; i < a.size1(); i++) {
		for (size_t j = 0; j < a.size2(); j++) {
			if (axis == 0) {
				r[i] += a(j, i);
			} else {
				r[i] += a(i, j);
			}
		}
	}

	return r;
}

ublas::matrix<float> lart::block_diag(std::vector<ublas::matrix<float>> a) {

	ublas::matrix<float> out;
	if (a.size() == 0)
		return out;

	std::vector<std::vector<size_t>> shapes;
	for (ublas::matrix<float> m : a) {
		std::vector<size_t> tmp = {m.size1(), m.size2()};
		shapes.push_back(tmp);
	}

	out = ublas::zero_matrix<float>(shapes[0][0] + shapes[0][1]);
	float r = 0;
	float c = 0;
	for (size_t i = 0; i < a.size(); ++i) {
		size_t rr = shapes[i][0];
		size_t cc = shapes[i][1];
		ublas::subrange(out, r, rr, c, cc) =  a[i];
		r += rr;
		c += cc;
	}

	return out;
}



}

