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
using KokkosDeviceMix = Kokkos::Device<HostExSpace, MemSpace>;
using HostKokkosDevice = Kokkos::Device<HostExSpace, HostMemSpace>;
int main(int, char**)
{
  Kokkos::initialize();
  {
  Kokkos::ViewTest<float> tasd{};
  int* num = static_cast<int*>(Kokkos::kokkos_malloc<MemSpace>(3 * sizeof(int)));
  num[0] = 1; num[1] = 1; num[2] = 0;
  Kokkos::View<int*, KokkosDevice> num_v(num, 3);
  //Kokkos::View<int*, KokkosDevice>* num_v(num, 3);
  std::cout << "View is trivially copyable:" << 
               std::is_trivially_copyable<decltype(tasd)>::value <<
               std::endl;
  std::cout << "View is std layout:" << 
               std::is_standard_layout<decltype(tasd)>::value <<
               std::endl;
                                                
                                                
  //auto q = Kokkos::Experimental::Impl::SYCLInternal::m_queue;
  auto num_vv = &num_v;
  int* num_vptr = num_v.data();
  auto lam = [=](const int idx) {
    //num[2] = num[0] + num[1];
    //(*num_vv)(2) = (*num_vv)(idx) + (*num_vv)(idx);
    //num_vptr[2] = num_vptr[idx] + num_vptr[idx];
    num_v(2) = num_v(idx) + num_v(idx);
  };

  //auto parfor = Kokkos::Impl::ParallelFor<decltype(lam), ExSpace>(lam, Kokkos::RangePolicy<ExSpace>(ExSpace(), 0, 2));
  //auto rpol = Kokkos::RangePolicy<ExSpace>(ExSpace(), 0, 2);
  Kokkos::parallel_for(2, lam);
  std::cout << "1 + 1 = " << num_v(2) << std::endl;
  }
  Kokkos::finalize();
  return 0;
}

