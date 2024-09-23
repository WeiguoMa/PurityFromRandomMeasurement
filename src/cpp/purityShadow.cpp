//
// Created by Weiguo Ma on 2024/9/23.
//

#include <vector>

using namespace std;

float tau(const int &u, const int &u_prime, const int &b, const int &b_prime) {
    if (u != u_prime) {
        return 0.5;
    } else {
        if (b == b_prime) {
            return 5;
        } else {
            return -4;
        }
    }
}

double purityEstimation(
        const vector<vector<int>> &measureOperations,
        const vector<vector<vector<int>>> &measureResults
) {
    size_t M = measureOperations.size();
    size_t K = measureResults[0].size();
    size_t N = measureResults[0][0].size();
    double factor = static_cast<int>(M) * (static_cast<int>(M) - 1) * pow(2.0, static_cast<int>(K));

    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) collapse(4)
    for (size_t m = 0; m < M; ++m) {
        for (size_t m_prime = m + 1; m_prime < M; ++m_prime) {
            for (size_t k = 0; k < K; ++k) {
                for (size_t k_prime = 0; k_prime < K; ++k_prime) {
                    double local_prod = 1.0;
                    for (size_t n = 0; n < N; ++n) {
                        local_prod *= tau(
                                measureOperations[m][n], measureOperations[m_prime][n],
                                measureResults[m][k][n], measureResults[m_prime][k_prime][n]
                        );
                        sum += local_prod;
                    }
                }
            }
        }
    }

    return (2.0 * sum) / factor;
}