//
// Created by Weiguo Ma on 2024/9/16.
//
#include "ShadowState.h"
#include "platformConfig.h"

namespace py = pybind11;

std::size_t matrix_hash::operator()(const std::pair<int, int> &p) const {
    std::size_t seed = 0;
    seed ^= std::hash<int>{}(p.first) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= std::hash<int>{}(p.second) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
}


ShadowState::ShadowState(const int &qnumber) {
    hilbert_dimension = static_cast<int>(pow(2, qnumber));
    precomputeAll();
}


void ShadowState::precomputeAll() {
    #pragma omp parallel for
    for (size_t i = 0; i < _pauliBases.size(); ++i) {
        for (size_t j = 0; j < _bases.size(); ++j) {
            MatrixXcd ketResult = _pauliBases[i].adjoint() * _bases[j];
            MatrixXcd rhoResult = ketResult * ketResult.adjoint();
            precomputedResults[{i, j}] = rhoResult;
        }
    }
}


template<typename MatrixType>
MatrixType ShadowState::kroneckerProduct(const MatrixType &A, const MatrixType &B) {
    const int rowsA = A.rows();
    const int colsA = A.cols();
    const int rowsB = B.rows();
    const int colsB = B.cols();

    MatrixType result(rowsA * rowsB, colsA * colsB);

    #pragma omp parallel for
    for (size_t i = 0; i < rowsA; ++i) {
        for (size_t j = 0; j < colsA; ++j) {
            result.block(i * rowsB, j * colsB, rowsB, colsB)
            = A(i, j) * B;
        }
    }
    return result;
}


template<typename MatrixType>
MatrixType ShadowState::kroneckerProduct(const std::vector<MatrixType> &matrixList) {
    MatrixType result = matrixList[0];
    for (size_t i = 1; i < matrixList.size(); ++i) {
        result = kroneckerProduct(result, matrixList[i]);
    }
    return result;
}


MatrixXcd ShadowState::measureResult2state(
        const vector<int> &measureOperation,
        const vector<vector<int>> &measureResult
) {
    MatrixXcd sum_measurements_state = MatrixXcd::Zero(hilbert_dimension, hilbert_dimension);

    vector<MatrixXcd> _state;
    _state.reserve(static_cast<int>(measureOperation.size()));

    for (vector<int> measureResult_per: measureResult) {
        _state.clear();
        for (size_t idx = 0; idx < measureOperation.size(); ++idx) {
            _state.emplace_back(
                    3 * precomputedResults[{measureOperation[idx], measureResult_per[idx]}] - MatrixXcd::Identity(2, 2)
            );
        }
        sum_measurements_state += kroneckerProduct(_state);
    }
    return sum_measurements_state / static_cast<int>(measureResult.size());
}


MatrixXcd ShadowState::stateEstimation(
        const vector<vector<int>> &measureOperations,
        const vector<vector<vector<int>>> &measureResults
) {
    MatrixXcd sumMatrix = MatrixXcd::Zero(hilbert_dimension, hilbert_dimension);
    #pragma omp parallel
    {
        MatrixXcd localSumMatrix = MatrixXcd::Zero(sumMatrix.rows(), sumMatrix.cols());

    #pragma omp for
        for (size_t idx = 0; idx < measureOperations.size(); ++idx) {
            localSumMatrix += ShadowState::measureResult2state(
                    measureOperations[idx],
                    measureResults[idx]
            );
        }

    #pragma omp critical
        {
            sumMatrix += localSumMatrix;    // Might increase the overhead
        }
    }

    return sumMatrix / static_cast<int>(measureOperations.size());
}


const vector<MatrixXcd> ShadowState::_pauliBases = {
        (MatrixXcd(2, 2) << 1, 1, 1, -1).finished() / sqrt(2.0),
        (MatrixXcd(2, 2) << 1, cd(0, -1), 1, cd(0, 1)).finished() / sqrt(2.0),
        MatrixXcd::Identity(2, 2)
};

const vector<MatrixXcd> ShadowState::_bases = {
        (MatrixXcd(2, 1) << 1, 0).finished(),
        (MatrixXcd(2, 1) << 0, 1).finished()
};


PYBIND11_MODULE(ShadowState_backend, m) {
    m.doc() = "ShadowState backend module";
    py::class_<ShadowState>(m, "ShadowState")
            .def(py::init<const int &>())
            .def("stateEstimation", &ShadowState::stateEstimation);
}