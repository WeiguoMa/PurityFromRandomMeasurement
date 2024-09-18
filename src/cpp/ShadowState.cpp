//
// Created by Weiguo Ma on 2024/9/16.
//
#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <vector>
#include <map>
#include <string>

using namespace Eigen;
using namespace std;

typedef complex<double> cd;

struct matrix_hash {
    std::size_t operator()(const std::pair<int, int> &p) const {
        std::size_t seed = 0;
        seed ^= std::hash<int>{}(p.first) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>{}(p.second) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

class ShadowState {
private:
    static const vector<MatrixXcd> _bases;
    static const vector<MatrixXcd> _pauliBases;
    unordered_map<pair<int, int>, MatrixXcd, matrix_hash> precomputedResults;

    void precomputeAll() {
        for (int i = 0; i < _pauliBases.size(); ++i) {
            for (int j = 0; j < _bases.size(); ++j) {
                MatrixXcd ketResult = _pauliBases[i].adjoint() * _bases[j];
                MatrixXcd rhoResult = ketResult * ketResult.adjoint();
                precomputedResults[{i, j}] = rhoResult;
            }
        }
    }

    template<typename MatrixType>
    static MatrixType kroneckerProduct(const MatrixType &A, const MatrixType &B) {
        const int rowsA = A.rows();
        const int colsA = A.cols();
        const int rowsB = B.rows();
        const int colsB = B.cols();

        MatrixType result(rowsA * rowsB, colsA * colsB);

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < rowsA; ++i) {
            for (int j = 0; j < colsA; ++j) {
                result.block(i * rowsB, j * colsB, rowsB, colsB) = A(i, j) * B;
            }
        }
        return result;
    }

    template<typename MatrixType>
    static MatrixType kroneckerProduct(const std::vector<MatrixType> &matrixList) {
        MatrixType result = matrixList[0];
        for (size_t i = 1; i < matrixList.size(); ++i) {
            result = kroneckerProduct(result, matrixList[i]);
        }
        return result;
    }

public:
    MatrixXcd measureResult2state(
            const vector<int> &measureOperation,
            const vector<int> &measureResult
    ) {
        vector<MatrixXcd> _state;
        _state.reserve(measureOperation.size());

        for (size_t idx = 0; idx < measureOperation.size(); ++idx) {
            _state.emplace_back(
                    3 * precomputedResults[{measureOperation[idx], measureResult[idx]}] - Matrix2cd::Identity()
            );
        }
        return kroneckerProduct(_state);
    }

    MatrixXcd stateEstimation(
            const vector<vector<int>> &measureOperations,
            const vector<vector<int>> &measureResults
    ) {
        MatrixXcd sumMatrix = MatrixXcd::Zero(precomputedResults.begin()->second.rows(),
                                              precomputedResults.begin()->second.cols());
        for (size_t idx = 0; idx < measureOperations.size(); ++idx) {
            sumMatrix += measureResult2state(measureOperations[idx], measureResults[idx]);
        }
        return sumMatrix / sumMatrix.size();
    }

    ShadowState() {
        precomputeAll();
    }
};

const vector<MatrixXcd> ShadowState::_pauliBases = {
        (MatrixXcd(2, 2) << 1, 1, -1, 1).finished() / sqrt(2.0),
        (MatrixXcd(2, 2) << 1, cd(0, -1), cd(0, 1), 1).finished() / sqrt(2.0),
        (MatrixXcd(2, 2) << 1, 0, 0, 1).finished()
};

const vector<MatrixXcd> ShadowState::_bases = {
        (MatrixXcd(1, 2) << 1, 0).finished(),
        (MatrixXcd(1, 2) << 0, 1).finished()
};