#include <CL/sycl.hpp>
#include <stdlib.h>
#include <iostream>
#include "Kokkos_Core.hpp"
using namespace cl::sycl;

using MemSpace = Kokkos::Experimental::SYCLHostUSMSpace;
using ExSpace = Kokkos::Experimental::SYCL::execution_space;
using KokkosDevice = Kokkos::Device<ExSpace, MemSpace>;
using CPUSpace = Kokkos::HostSpace;
using HostKokkosDevice = Kokkos::Device<CPUSpace, MemSpace>;

int main(int, char**)
{
  Kokkos::initialize();
  {
  Kokkos::View<int*, KokkosDevice> num_v("label", 3);
  num_v(0) = 1; num_v(1) = 1; num_v(2) = 0;
  auto lam = [=](const int idx) {
    num_v(2) = num_v(idx) + sqrt(num_v(idx));
  };

  Kokkos::parallel_for(2, lam);
  std::cout << "1 + 1 = " << num_v(2) << std::endl;
  }
  Kokkos::finalize();
  return 0;
}

