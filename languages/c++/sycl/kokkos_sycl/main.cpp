//#include <CL/sycl.hpp>
#include "Kokkos_Core.hpp"
#include <stdlib.h>
#include <iostream>
//using namespace cl::sycl;

using HostSpace = Kokkos::HostSpace;

using MemSpace = Kokkos::Experimental::SYCLHostUSMSpace;
using ExSpace = Kokkos::Experimental::SYCL;
//using MemSpace = Kokkos::Experimental::OpenMPTargetSpace;
//using ExSpace = Kokkos::Experimental::OpenMPTarget;
//using MemSpace = Kokkos::HostSpace;
//using ExSpace = Kokkos::Serial;
using KokkosDevice = Kokkos::Device<ExSpace, MemSpace>;

int main(int, char**)
{
  int num[3];
  Kokkos::initialize();
  {
  //Kokkos::View<int*, KokkosDevice> num_v("label", 3);
  Kokkos::View<int*, KokkosDevice> num_v(num, 3);
  Kokkos::View<int*, HostSpace> num_h(num, 3);
  num_v(0) = 1; num_v(1) = 1; num_v(2) = 0;
  auto lam = [=](const int idx) {
    //num_v(2) = num_v(idx) + (num_v(idx));
    num_v(2) = 99;
  };

  Kokkos::parallel_for(2, lam);
  Kokkos::fence();
  std::cout << "1 + 1 = " << num_v(2) << std::endl;
  Kokkos::deep_copy(num_h, num_v);
  }
  Kokkos::finalize();
  std::cout << "1 + 1 = " << num[2] << std::endl;
  return 0;
}

