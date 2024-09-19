//
// Created by Weiguo Ma on 2024/9/18.
//

#ifndef PURITYFROMSHADOW_SHADOWSTATE_H
#define PURITYFROMSHADOW_SHADOWSTATE_H

#include <Eigen/Dense>
#include <complex>
#include <vector>
#include <unordered_map>
#include <utility>

using namespace Eigen;
using namespace std;

typedef complex<double> cd;

struct matrix_hash {
    std::size_t operator()(const std::pair<int, int> &p) const;
};

class ShadowState {
private:
    static const vector<MatrixXcd> _bases;
    static const vector<MatrixXcd> _pauliBases;
    int hilbert_dimension;
    unordered_map<pair<int, int>, MatrixXcd, matrix_hash> precomputedResults;

    void precomputeAll();

    template<typename MatrixType>
    static MatrixType kroneckerProduct(const MatrixType &A, const MatrixType &B);

    template<typename MatrixType>
    static MatrixType kroneckerProduct(const std::vector<MatrixType> &matrixList);

public:
    explicit ShadowState(const int &qnumber);

    MatrixXcd measureResult2state(const vector<int> &measureOperation, const vector<vector<int>> &measureResult);

    MatrixXcd
    stateEstimation(const vector<vector<int>> &measureOperations, const vector<vector<vector<int>>> &measureResults);
};

#endif //PURITYFROMSHADOW_SHADOWSTATE_H
