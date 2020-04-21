#include <CL/sycl.hpp>
#include <stdlib.h>
#include <iostream>
#include "Kokkos_Core.hpp"
using namespace cl::sycl;

using MemSpace = Kokkos::Experimental::SYCLHostUSMSpace;
using ExSpace = Kokkos::Experimental::SYCL::execution_space;
using HostMemSpace = Kokkos::HostSpace;
using HostExSpace = Kokkos::Serial;
using KokkosDevice = Kokkos::Device<ExSpace, MemSpace>;
using HostKokkosDevice = Kokkos::Device<HostExSpace, HostMemSpace>;
int main(int, char**)
{
  Kokkos::initialize();
  {
  int* num = static_cast<int*>(Kokkos::kokkos_malloc<MemSpace>(3 * sizeof(int)));
  num[0] = 1; num[1] = 1; num[2] = 0;
  Kokkos::View<int*, KokkosDevice> num_v(num, 3);
  std::cout << "View is trivially copyable:" << 
               std::is_trivially_copyable<decltype(num_v)>::value <<
               std::endl;
  std::cout << "View is std layout:" << 
               std::is_standard_layout<decltype(num_v)>::value <<
               std::endl;
                                                
                                                
  auto num_vptr = num_v.data();
  auto lam = [=](const int idx) {
    num_vptr[2] = num_vptr[idx] + num_vptr[idx];
  };
  Kokkos::parallel_for(2, lam);
  std::cout << "1 + 1 = " << num[2] << std::endl;
  }
  Kokkos::finalize();
  return 0;
}

