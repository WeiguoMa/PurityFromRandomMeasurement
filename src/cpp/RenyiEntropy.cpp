//
// Created by Weiguo Ma on 2024/9/18.
//
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>

using namespace std;

int countBits(uint64_t x) {
    return __builtin_popcountll(x);
}

int hammingDistance(const vector<uint64_t>& v1, const vector<uint64_t>& v2, size_t numBits) {
    int distance = 0;
    size_t numBlocks = v1.size();
    for (size_t i = 0; i < numBlocks; ++i) {
        distance += countBits(v1[i] ^ v2[i]);
    }

    size_t extraBits = numBits % 64;
    if (extraBits > 0) {
        uint64_t mask = (1ULL << extraBits) - 1;
        distance += countBits((v1.back() ^ v2.back()) & mask);
    }

    return distance;
}

class RenyiEntropy {
private:
    size_t N; // Number of Qubits
    size_t K; // Number of Measurements with given U
    size_t M; // Number of Us

    vector<vector<vector<uint64_t>>> measurementResultsBitset;

    [[nodiscard]] vector<uint64_t> compressVector(const vector<int>& binaryVector) const {
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
    explicit RenyiEntropy(const vector<vector<vector<int>>>& measurementResults) {
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

    double calculateP2() {
        double sum = 0.0;

        for (size_t m = 0; m < M; ++m) {
            for (size_t k = 0; k < K; ++k) {
                for (size_t k_prime = k + 1; k_prime < K; ++k_prime) {
                    int dist = hammingDistance(measurementResultsBitset[m][k], measurementResultsBitset[m][k_prime], N);
                    sum += 2 * pow(-2, -dist);  // k/k' is the same as k'/k
                }
            }
        }

        return (pow(2.0, static_cast<double>(N)) /
                (static_cast<double>(M) * static_cast<double>(K) * static_cast<double>(K - 1))) * sum;

    }

    double calculateRenyiEntropy(){
        return -log2(calculateP2());
    }
};
