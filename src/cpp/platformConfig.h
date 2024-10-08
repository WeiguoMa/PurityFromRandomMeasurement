//
// Created by Weiguo Ma on 2024/9/20.
//

#ifndef PURITYFROMSHADOW_PLATFORMCONFIG_H
#define PURITYFROMSHADOW_PLATFORMCONFIG_H

#if defined(_WIN32) || defined(_WIN64)
    #include <bit>
    #include <random>

    using randomGenerator = std::mt19937;

    inline int countBits(uint64_t x) {
        return std::popcount(x);
    }
#else
    #include "pcg/pcg_random.hpp"

    using randomGenerator = pcg32;

    inline int countBits(int x) {
        return __builtin_popcountll(x);
    }

#endif

#endif //PURITYFROMSHADOW_PLATFORMCONFIG_H
