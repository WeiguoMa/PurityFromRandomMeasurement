//
// Created by Weiguo Ma on 2024/9/18.
//
#include "ShadowState.h"
#include "fwht.cpp"
#include "purityShadow.cpp"

using namespace std;
namespace py = pybind11;


class RenyiEntropy_backend {
private:
    size_t N; // Number of Qubits
    size_t K; // Number of Measurements with given U
    size_t M; // Number of Us

    ShadowState shadowState;
    vector<vector<int>> measurementScheme;
    vector<vector<vector<int>>> measurementResults;
    vector<vector<vector<uint64_t>>> measurementResultsBitset;

public:
    RenyiEntropy_backend(const vector<vector<int>> &measurementScheme,
                         const vector<vector<vector<int>>> &measurementResults)
            : measurementScheme(measurementScheme), measurementResults(measurementResults),
              shadowState(static_cast<int>(measurementResults[0][0].size())) {
        M = measurementResults.size();
        K = measurementResults[0].size();
        N = measurementResults[0][0].size();
    }

    double calculateP2_Hamming() {
        double sum = 0.0;
        FWHTSolver fwhtSolver(static_cast<int>(N));

        #pragma omp parallel for reduction(+:sum)
        for (size_t m = 0; m < M; ++m) {
            sum += fwhtSolver.computeSumS(measurementResults[m]);
        }
        return (pow(2.0, static_cast<double>(N)) /
                (static_cast<double>(M) * static_cast<double>(K) * static_cast<double>(K - 1))) * sum;

    }

    double calculateP2_ClassicalShadow() {
        double sum = 0.0;

        vector<MatrixXcd> rhoMatrices(M);
        #pragma omp parallel for
        for (size_t m = 0; m < M; ++m) {
            rhoMatrices[m] = shadowState.stateEstimation(
                    {measurementScheme[m]},
                    {measurementResults[m]}
            );
        }

        #pragma omp parallel for reduction(+:sum) collapse(2)
        for (size_t m = 0; m < M; ++m) {
            for (size_t m_prime = m + 1; m_prime < M; ++m_prime) {
                MatrixXcd product = rhoMatrices[m] * rhoMatrices[m_prime];      // rho^{(m)} * rho^{(m')}
                sum += product.trace().real();
            }
        }
        return (2.0 * sum) / (static_cast<int>(M) * (static_cast<int>(M) - 1));  // times 2, m != m'
    };

//    double calculateP2_ClassicalShadow() {
//        double purity = purityEstimation(measurementScheme, measurementResults);
//        return purity;
//    };

    double calculateRenyiEntropy(bool cs) {
        if (cs) {
            double renyi2_cs = calculateP2_ClassicalShadow();
            return -log2(renyi2_cs);
        } else {
            double renyi2_hamming = calculateP2_Hamming();
            return -log2(renyi2_hamming);
        }
    }
};

PYBIND11_MODULE(RenyiEntropy_backend, m) {
    py::class_<RenyiEntropy_backend>(m, "RenyiEntropy_backend")
            .def(py::init<const vector<vector<int>> &, const vector<vector<vector<int>>> &>(),
                 py::arg("measurementScheme"), py::arg("measurementResults"))
            .def("calculateRenyiEntropy", &RenyiEntropy_backend::calculateRenyiEntropy,
                 py::arg("cs") = false);
}