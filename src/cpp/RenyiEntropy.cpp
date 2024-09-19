//
// Created by Weiguo Ma on 2024/9/18.
//
#include <pybind11/pybind11.h>
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <unordered_map>
#include "ShadowState.h"

using namespace std;
namespace py = pybind11;

int countBits(uint64_t x) {
    return __builtin_popcountll(x);
}

int hammingDistance(const vector<uint64_t> &v1, const vector<uint64_t> &v2, size_t numBits) {
    int distance = 0;
    size_t numBlocks = v1.size();

    for (size_t i = 0; i < numBlocks - 1; ++i) {
        distance += countBits(v1[i] ^ v2[i]);
    }

    size_t extraBits = numBits % 64;
    if (extraBits > 0) {
        uint64_t mask = (1ULL << extraBits) - 1;  // Mask for valid bits in the last block
        distance += countBits((v1.back() ^ v2.back()) & mask);
    } else {
        distance += countBits(v1.back() ^ v2.back());
    }

    return distance;
}


class RenyiEntropy_backend {
private:
    size_t N; // Number of Qubits
    size_t K; // Number of Measurements with given U
    size_t M; // Number of Us

    ShadowState shadowState;
    vector<vector<int>> measurementScheme;
    vector<vector<vector<int>>> measurementResults;
    vector<vector<vector<uint64_t>>> measurementResultsBitset;

    [[nodiscard]] vector<uint64_t> compressVector(const vector<int> &binaryVector) const {
        size_t numBlocks = (N + 63) / 64;
        vector<uint64_t> compressed(numBlocks, 0);

        for (size_t i = 0; i < N; ++i) {
            size_t blockIndex = i / 64;
            size_t bitIndex = i % 64;
            if (binaryVector[i] == 1) {
                compressed[blockIndex] |= (1ULL << bitIndex);
            }
        }

        return compressed;
    }

public:
    RenyiEntropy_backend(const vector<vector<int>> &measurementScheme,
                         const vector<vector<vector<int>>> &measurementResults)
            : measurementScheme(measurementScheme), measurementResults(measurementResults),
              shadowState(static_cast<int>(measurementResults[0][0].size())) {
        M = measurementResults.size();
        K = measurementResults[0].size();
        N = measurementResults[0][0].size();

        for (size_t m = 0; m < M; ++m) {
            vector<vector<uint64_t>> compressedSet;
            for (size_t k = 0; k < K; ++k) {
                compressedSet.push_back(compressVector(measurementResults[m][k]));
            }
            measurementResultsBitset.push_back(compressedSet);
        }
    }

    double calculateP2_Hamming() {
        double sum = 0.0;

        for (size_t m = 0; m < M; ++m) {
            for (size_t k = 0; k < K; ++k) {
                for (size_t k_prime = k + 1; k_prime < K; ++k_prime) {
                    int dist = hammingDistance(
                            measurementResultsBitset[m][k],
                            measurementResultsBitset[m][k_prime],
                            N
                    );
                    sum += 2 * pow(-2, -dist);  // k/k' is the same as k'/k
                }
            }
        }
        return (pow(2.0, static_cast<double>(N)) /
                (static_cast<double>(M) * static_cast<double>(K) * static_cast<double>(K - 1))) * sum;

    }

    double calculateP2_ClassicalShadow() {
        double sum = 0.0;

        vector<MatrixXcd> rhoMatrices;
        for (size_t m = 0; m < M; ++m) {
            MatrixXcd rho_m = shadowState.stateEstimation(
                    {measurementScheme[m]}, {measurementResults[m]}
            );
            rhoMatrices.push_back(rho_m);
        }

        for (size_t m = 0; m < M; ++m) {
            for (size_t m_prime = m + 1; m_prime < M; ++m_prime) {
                MatrixXcd product = rhoMatrices[m] * rhoMatrices[m_prime];  // rho^{(m)} * rho^{(m')}
                sum += product.trace().real();
            }
        }
        return (2.0 * sum) / (static_cast<int>(M) * (static_cast<int>(M) - 1));  // times 2, m != m'
    };

    double calculateRenyiEntropy(bool cs) {
        if (cs) {
            return -log2(calculateP2_ClassicalShadow());
        } else {
            return -log2(calculateP2_Hamming());
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