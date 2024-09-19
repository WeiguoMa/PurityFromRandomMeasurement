#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <random>
#include <vector>
#include <map>
#include <string>

namespace py = pybind11;
using namespace Eigen;
using namespace std;

typedef complex<double> cd;

class FakeSampler_backend {
private:
    vector<string> proj_basis;
    vector<MatrixXcd> _bases = {
            (MatrixXcd(2, 2) << 1, 1, -1, 1).finished() / sqrt(2.0),
            (MatrixXcd(2, 2) << 1, cd(0, -1), cd(0, 1), 1).finished() / sqrt(2.0),
            MatrixXcd::Identity(2, 2)
    };

    std::random_device rd;
    std::mt19937 gen;

    template<typename MatrixType>
    static MatrixType kroneckerProduct(const MatrixType& A, const MatrixType& B) {
        const int rowsA = A.rows();
        const int colsA = A.cols();
        const int rowsB = B.rows();
        const int colsB = B.cols();

        MatrixType result(rowsA * rowsB, colsA * colsB);

        for (int i = 0; i < rowsA; ++i) {
            for (int j = 0; j < colsA; ++j) {
                result.block(i * rowsB, j * colsB, rowsB, colsB) = A(i, j) * B;
            }
        }
        return result;
    }
public:
    explicit FakeSampler_backend(vector<string>& basis): proj_basis(basis), gen(rd()) {}

    vector<vector<int>> fakeSampling_dm(const py::array_t<std::complex<double>> &dm_array,
                                const int &measure_times,
                                std::vector<int> &measurement_orientation);
};


vector<vector<int>> FakeSampler_backend::fakeSampling_dm(const py::array_t<std::complex<double>> &dm_array,
                                                 const int &measure_times,
                                                 std::vector<int> &measurement_orientation) {
    auto dm_buf = dm_array.request();

    const size_t rows = dm_buf.shape[0];
    const size_t cols = dm_buf.shape[1];

    Eigen::Map<MatrixXcd> dm(
            static_cast<std::complex<double>*>(dm_buf.ptr),
            static_cast<int>(rows), static_cast<int>(cols)
            );

    MatrixXcd U_operations = _bases[measurement_orientation[0]];
    for (size_t i = 1; i < measurement_orientation.size(); ++i) {
        U_operations = kroneckerProduct(U_operations, _bases.at(measurement_orientation[i]));
    }

    MatrixXcd _operated_DM = U_operations * dm * U_operations.adjoint();

    std::vector<double> _prob_real;
    _prob_real.reserve(_operated_DM.rows());

    for (int i = 0; i < _operated_DM.rows(); ++i) {
        _prob_real.push_back(_operated_DM(i, i).real());
    }

    std::discrete_distribution<> dist(_prob_real.begin(), _prob_real.end());

    vector<vector<int>> results;
    results.reserve(measure_times);

    for (int t = 0; t < measure_times; ++t) {
        std::string _state = proj_basis[dist(gen)];

        vector<int> _state_eigenValue;
        _state_eigenValue.reserve(_state.size());
        for (char _value : _state) {
            _state_eigenValue.push_back(_value == '0' ? 0 : 1);
        }

        results.push_back(_state_eigenValue);
    }

    return results;
}

PYBIND11_MODULE(fakeSampler_backend, m) {
    py::class_<FakeSampler_backend>(m, "FakeSampler_backend")
            .def(py::init<vector<string>&>(), py::arg("proj_basis"))
            .def("fakeSampling_dm", &FakeSampler_backend::fakeSampling_dm,
                 py::arg("dm_array"), py::arg("measure_times"), py::arg("measurement_orientation"));
}
