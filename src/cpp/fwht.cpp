#include <vector>
#include <cmath>
#include <algorithm>
#include "platformConfig.h"

const int N = 4;
const int SIZE = 1 << N; // 2^N

void fwht(std::vector<double>& data, bool inverse) {
    int n = static_cast<int>(data.size());
    for (int len = 1; 2 * len <= n; len <<= 1) {
        for (int i = 0; i < n; i += 2 * len) {
            for (int j = 0; j < len; ++j) {
                double u = data[i + j];
                double v = data[i + j + len];
                data[i + j] = u + v;
                data[i + j + len] = u - v;
            }
        }
    }
    if (inverse) {
        for (auto& x : data) x /= n;
    }
}

double computeSumS(const std::vector<std::vector<int>>& S) {
    int K = static_cast<int>(S.size());
    std::vector<double> M(SIZE, 0.0);

    for (const auto& s : S) {
        int t = 0;
        for (int i = 0; i < N; ++i) {
            if (s[i]) t |= 1 << i;
        }
        M[t] += 1.0;
    }

    fwht(M, false);

    for (auto& x : M) x *= x;

    fwht(M, true);

    std::vector<double> c_D(N + 1, 0.0);
    for (int s = 0; s < SIZE; ++s) {
        int D = countBits(s); // Hamming weight
        c_D[D] += M[s];
    }

    double S_prime = 0.0;
    for (int D = 0; D <= N; ++D) {
        double term = c_D[D] * pow(-0.5, D);
        S_prime += term;
    }
    double S_final = S_prime - K;
    return S_final;
}
