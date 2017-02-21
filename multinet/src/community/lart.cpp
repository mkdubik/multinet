#include "community.h"
#include <boost/numeric/ublas/matrix_proxy.hpp>

namespace mlnet {


hash_set<ActorSharedPtr> lart::get_ml_community(
	MLNetworkSharedPtr mnet, uint32_t t, double eps, double gamma) {

	hash_set<ActorSharedPtr> actors;

	std::vector<ublas::matrix<double>> a = ml_network2adj_matrix(mnet);
	ublas::matrix<double> A = supraA(a, eps);
	ublas::matrix<double> D = diagA(A);
	ublas::matrix<double> P = ublas::prod(D, A);
	//prcheck(a, P);
	ublas::matrix<double> Pt = matrix_power(P, t);
	//std::cout << P << std::endl;
	Dmat(Pt, D, a.size());

	return actors;
}

std::vector<ublas::matrix<double>> lart::ml_network2adj_matrix(MLNetworkSharedPtr mnet) {

	size_t N = mnet->get_layers()->size();
	size_t M = mnet->get_actors()->size();

	std::vector<ublas::matrix<double>> adj(N);

	for (LayerSharedPtr l: *mnet->get_layers()) {
		ublas::matrix<double> m = ublas::zero_matrix<double>(M);

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

ublas::matrix<double> lart::Dmat(ublas::matrix<double> Pt, ublas::matrix<double> D, size_t L) {

	size_t N = Pt.size1();

	//std::cout << N << std::endl;
	ublas::matrix<double> Dt = ublas::zero_matrix<double>(Pt.size1(), Pt.size2());


	std::transform(D.data().begin(), D.data().end(), D.data().begin(),
		static_cast<double (*)(double)>(&std::sqrt));
	ublas::matrix<double> newP = ublas::element_prod(Pt, D);

	//std::cout << Pt << std::endl;
	//std::cout << D << std::endl;
	//std::cout << newP << std::endl;

	int i = 0;
	ublas::matrix<double> k = pairwise_distance(ublas::subrange(newP, (i*N), (i+1)*N, (i*N), ((i+1)*N)));
	//std:: cout << k << std::endl;

	//for (size_t i = 0; i < L; i++) {
		//ublas::subrange(Dmat, (i*N), (i+1)*N, (i*N), ((i+1)*N)) =
	//}

	return Dt;

}

void lart::printm(const ublas::matrix<double> &m)
{
	std::cout<<"[ ";
    for(unsigned i=0;i<m.size1();++i)
    {
        std::cout<<"[";
        for (unsigned j=0;j<m.size2();++j)
        {
            std::cout<<m(i,j)<<", ";
        }
        std::cout << "],";
    }
	std::cout<<" ]" ;

}


ublas::matrix<double> lart::pairwise_distance(ublas::matrix<double> X) {

	// Euclidean distance

	ublas::vector<double> XX = sum(ublas::element_prod(X, X), 1);

	//std::cout << XX << std::endl;
	//std::cout << X << std::endl;
	//printm(X);


	ublas::vector<double> YY = ublas::trans(XX);


	ublas::matrix<double> distances = ublas::element_prod(X, ublas::trans(X));

	std::transform(distances.data().begin(), distances.data().end(), distances.data().begin(),
		std::bind1st(std::multiplies<double>(), -2));

	for (size_t i = 0; i < X.size1(); i++) {
		for (size_t j = 0; j < X.size2(); j++) {
			if (i == j) {
				distances(i, j) = 0.0;
				continue;
			}
			distances(i, j) += XX(i);
			distances(i, j) += YY(i);
		}
	}

	std::transform(distances.data().begin(), distances.data().end(),
		distances.data().begin(), static_cast<double (*)(double)>(&std::sqrt));

	return distances;
}

ublas::matrix<double> lart::block_diag(std::vector<ublas::matrix<double>> a) {

	ublas::matrix<double> out;
	if (a.size() == 0)
		return out;

	out = ublas::zero_matrix<double>(a[0].size1() * a.size(), a[0].size2() * a.size());

	size_t r, c, rr, cc;
	r = 0;
	c = 0;
	rr = 0;
	cc = 0;

	for (size_t i = 0; i < a.size(); i++) {
		rr = r + a[i].size1();
		cc = c + a[i].size2();

		ublas::subrange(out, r, rr, c, cc) = a[i];

		r = rr;
		c = cc;
	}
	return out;
}


ublas::matrix<double> lart::supraA(std::vector<ublas::matrix<double>> a, double eps) {

	ublas::matrix<double> A = block_diag(a);

	for (size_t i = 0; i < A.size1(); ++i) {
		A(i, i) = eps;
	}

	size_t L = a.size();
	size_t N = a[0].size1();

	for (size_t i = 0; i  < L - 1; ++i) {
		for (size_t j = i + 1; j < L; ++j) {
			ublas::vector<double> d = sum(ublas::element_prod(a[i], a[j]), 0);

			ublas::matrix_vector_range<ublas::matrix<double> > tmp1(A,
				ublas::range(i * N, (i + 1) * N),
				ublas::range(j * N, (j + 1) * N));

			for (size_t k = 0; k < d.size(); ++k) {
				tmp1(k) = d[k] + eps;
			}

			ublas::matrix_vector_range<ublas::matrix<double> > tmp2(A,
				ublas::range(j * N, (j + 1) * N),
				ublas::range(i * N, (i + 1) * N));

			for (size_t k = 0; k < d.size(); ++k) {
				tmp2(k) = d[k] + eps;
			}
		}
	}

	return A;
}


ublas::matrix<double> lart::diagA(ublas::matrix<double> a) {

	ublas::matrix<double> A = ublas::zero_matrix<double>(a.size1(), a.size2());

	ublas::vector<double> d = sum(a, 0);

	//std::cout << "a " << a << std::endl;

	for (size_t i = 0; i < d.size(); i++) {
		if (d[i] > 0) {
			A(i, i) = 1 / d[i];
		} else {
			A(i, i) = 0;
		}
	}
	//std::cout << "-------" << std::endl;

	//std::cout << "A " << A << std::endl;

	return A;
}


void lart::prcheck(std::vector<ublas::matrix<double>> a,
			ublas::matrix<double> P) {

	//ublas::matrix<double> g = sum(a, 0);
	//TODO
}


/*ublas::matrix<double> lart::sum(std::vector<ublas::matrix<double>> a, int axis) {

	ublas::matrix<double> m = ublas::zero_matrix<double>(a[0].size1(), a[0].size2());

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
}*/

ublas::matrix<double> lart::matrix_power(ublas::matrix<double> m, uint32_t t) {

	if (t == 0) {
		return ublas::identity_matrix<double>(m.size1(), m.size2());
	}

	ublas::matrix<double> Dt(m);

	for (uint32_t i = 1; i < t; i++) {
		Dt = ublas::prod(Dt, m);
	}
	return Dt;
}

ublas::vector<double> lart::sum(ublas::matrix<double> a, int axis) {

	ublas::vector<double> r = ublas::zero_vector<double>(a.size1());

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




}

