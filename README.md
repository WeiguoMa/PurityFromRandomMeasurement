## Project can be compiled with following steps:
````
mkdir build
````
````
cd build
````
````
cmake -DUSE_OPENMP=OFF ..
````
````
make
````
This project is packaged by python with C++ core.

### Problems:
1. This project is not universal for Windows now, some hard code should be done to make everything works;
2. I use OpenMP to accelerate the computation in some cases, but they are currently naive and not optimized for POOR Mac,
which may cause less efficiency than the serial version. So I turn off the OpenMP by default (But it is tested to be well
on Windows platform).
3. In some cases, OpenMP causes unexpected numerical errors.