//
// Created by Weiguo Ma on 2024/9/20.
//

#ifndef PURITYFROMSHADOW_PLATFORMCONFIG_H
#define PURITYFROMSHADOW_PLATFORMCONFIG_H

#if defined(_WIN32) || defined(_WIN64)
    #include <bit>

    using LoopIndexType = int;
    using randomGenerator = std::mt19937;

    inline int countBits(uint64_t x) {
        return std::popcount(x);
    }
#else
    #include "pcg/pcg_random.hpp"

    using LoopIndexType = size_t;
    using randomGenerator = pcg32;

    inline int countBits(uint64_t x) {
        return __builtin_popcountll(x);
    }

#endif

#endif //PURITYFROMSHADOW_PLATFORMCONFIG_H
