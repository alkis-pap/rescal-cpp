// Rescal Tensor Factorization using Eigen
// Author: Alkis Papadopoulos

#include <iostream>
#include <cmath>
#include <complex>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Spectre/GenEigsSolver.h>
#include <Spectre/MatOp/SparseGenMatProd.h>

#ifndef NO_PYTHON
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#endif

using namespace Eigen;
using namespace Spectra;

typedef SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;


// Vector of triplets.
class EdgeList {
public:
    EdgeList(const int n_relations, const int n_nodes) : n_relations(n_relations), n_nodes(n_nodes), edges(n_relations) {}

    void add_edge(const int r, int i, int j) {
        edges[r].emplace_back(T(i, j, 1.0));
    }

    std::vector<SpMat> get_tensor() const {
        std::vector<SpMat> result;
        for (size_t i = 0; i < edges.size(); i++) {
            result.emplace_back(SpMat(n_nodes, n_nodes));
            result.back().setFromTriplets(edges[i].begin(), edges[i].end());
        }
        return result;
    }

    // Clear and de-allocate memory
    void clear() {
        edges.clear();
        edges.shrink_to_fit();
    }

private:
    int n_relations, n_nodes;
    std::vector<std::vector<T> > edges;
};


void update_A(const std::vector<SpMat>& X, MatrixXd& A, const std::vector<MatrixXd>& R, const double lambda_A) {
    const int n = A.rows();
    const int rank = A.cols();
    MatrixXd F = MatrixXd::Zero(n, rank);
    MatrixXd E = MatrixXd::Zero(rank, rank);
    const MatrixXd AtA = A.transpose() * A;
    for (size_t i = 0; i < X.size(); i++) {
        F += X[i] * A * R[i].transpose() + X[i].transpose() * A * R[i];
        E += R[i] * AtA * R[i].transpose() + R[i].transpose() * AtA * R[i];
    }
    E.transposeInPlace();
    E += lambda_A * MatrixXd::Identity(rank, rank);
    HouseholderQR<MatrixXd> dec(E);
    A = dec.solve(F.transpose()).transpose();
}

void update_R(const std::vector<SpMat>& X, const MatrixXd& A, std::vector<MatrixXd>& R, const double lambda_R) {
    BDCSVD<MatrixXd> svd = A.bdcSvd(ComputeThinU | ComputeThinV);
    const MatrixXd& U = svd.matrixU();
    const MatrixXd& V = svd.matrixV();
    const VectorXd& S = svd.singularValues();
    const size_t size = S.rows();
    ArrayXXd Shat(size, size);
    double s;
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            s = S(i) * S(j);
            Shat(i, j) = s / (s * s + lambda_R);
        }
    }
    for (size_t i = 0; i < X.size(); i++) {
        R[i] = V * (Shat * (U.transpose() * X[i] * U).array()).matrix() * V.transpose();
    }
}

double _compute_fit
(
    const std::vector<SpMat>& X, 
    const MatrixXd& A, 
    const std::vector<MatrixXd>& R, 
    const double lambda_A, 
    const double lambda_R,
    const std::vector<double> norm_X
)
{
    double f = lambda_A * std::pow(A.norm(), 2);
    for (size_t i = 0; i < X.size(); i++) {
        f += (X[i] - A * R[i] * A.transpose()).squaredNorm() / norm_X[i] + lambda_R * R[i].squaredNorm();
    }
    return f;
}

class Rescal {
private:
    MatrixXd A;
    std::vector<MatrixXd> R;
    bool set_a = false;

public:
    Rescal() {}

    void set_A(MatrixXd A) { this->A = A; set_a = true; }

    // Return references to avoid copies
    MatrixXd &get_A() { return A; }
    MatrixXd &get_R_slice(int i) { return R[i]; }

    double als
    (
        const EdgeList &edge_list,
        const int rank, 
        const double lambda_A, 
        const double lambda_R, 
        const bool init_random = false, 
        const size_t max_iter = 250,
        const int compute_fit = 0,
        const double e = 1e-4,
        const int verbose = 0
    )
    {
        std::vector<SpMat> X = edge_list.get_tensor();
        int n = X[0].rows();
        int n_rel = X.size();

        R.clear();
        R.shrink_to_fit();
        R.reserve(n_rel);
        for (size_t i = 0; i < X.size(); i++)
            R.emplace_back(MatrixXd::Zero(rank, rank));

        if (!set_a) {
            if (init_random) {
                A = MatrixXd::Random(n, rank).cwiseAbs();
            }
            else {
                A = MatrixXd::Zero(n, rank);
                SpMat S(n, n);
                for (const auto& x : X) {
                    S += SpMat(x.transpose()) + x;
                }
                
                //Copute eigenvectors
                SparseGenMatProd<double> op(S);
                GenEigsSolver< double, LARGEST_MAGN, SparseGenMatProd<double> > eigs(&op, rank, std::min(2 * rank  + 1, n));
                eigs.init();
                eigs.compute();

                A = eigs.eigenvectors().real();

                // MatrixXd evecs = eigs.eigenvectors().real();
                // std::vector<size_t> index_vec;
                // for (Eigen::Index i = 0; i != evecs.cols(); ++i) { 
                //     index_vec.push_back(i); 
                // }
                // // sort by eigenvalue
                // std::sort(
                //     index_vec.begin(), index_vec.end(),
                //     [&] (const size_t lhs, const size_t rhs) {
                //         return evecs(0, lhs) < evecs(0, rhs);
                //     }
                // );
                
                // for (size_t i = 0; i < index_vec.size(); i++) {
                //     A.col(i) = evecs.col(index_vec[i]);
                // }
            }
        }
        
        update_R(X, A, R, lambda_R);
        std::vector<double> norm_X;
        std::cout << "norm_X" << std::endl;
        for (const auto& x : X) {
            double sn = x.squaredNorm();
            norm_X.push_back(sn);
            std::cout << sn << std::endl;
        }
        double fit = 0, d_fit = 0;
        for (size_t i = 0; i < max_iter; i ++) {
            if (verbose > 0)
                std::cout << "Iteration " << i << std::endl;
            if(verbose > 1) {
                std::cout << A << std::endl;
                std::cout << R[0] << std::endl;
            }

            update_A(X, A, R, lambda_A);
            update_R(X, A, R, lambda_R);

            //compute fit
            if (compute_fit && i % compute_fit <= 1) {
                d_fit = _compute_fit(X, A, R, lambda_A, lambda_R, norm_X) - fit;
                fit += d_fit;
                if (verbose) std::cout << "fit = " << fit << std::endl;
                if (i > 0 && std::fabs(d_fit) < e)
                    return fit;
            }
        }
        fit = _compute_fit(X, A, R, lambda_A, lambda_R, norm_X);
        return fit;
    }
};

#ifndef NO_PYTHON

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(rescal, m) {
    m.doc() = "rescal"; // optional module docstring

    py::class_<EdgeList>(m, "EdgeList")
        .def(py::init<const int, const int>())
        .def("add_edge", &EdgeList::add_edge)
        .def("clear", &EdgeList::clear);

    py::class_<Rescal>(m, "Rescal")
        .def(py::init<>())
        .def("als", &Rescal::als, 
            "X"_a, 
            "rank"_a, 
            "lambda_A"_a, 
            "lambda_R"_a, 
            "init_random"_a = false, 
            "max_iter"_a = 250, 
            "compute_fit"_a = 0, 
            "e"_a = 1e-4,
            "verbose"_a = 0)
        .def("get_A", &Rescal::get_A, py::return_value_policy::reference_internal)
        .def("set_A", &Rescal::set_A)
        .def("get_R_slice", &Rescal::get_R_slice, py::return_value_policy::reference_internal);
}

#endif