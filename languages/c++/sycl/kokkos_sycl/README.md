# Kokkos SYCL Test
Simple test case for Kokkos with SYCL backend
Tracking https://github.com/nliber/kokkos/tree/sycl-nliber

## Compile and Test
Assumes Kokkos installation at $KOKKOS_HOME.
Set your C++ compiler to $CC.

````
export KOKKOS_HOME=/path/to/built/kokkos
export CC=clang++
make
./intel
````

