//
// Created by Weiguo Ma on 2024/9/22.
//
#include <iostream>
#include <omp.h>

int main() {
    std::cout << "OpenMP version" << _OPENMP << std::endl;
    int num_procs = omp_get_num_procs();
    std::cout << "Number of processors: " << num_procs << std::endl;
    return 0;
}