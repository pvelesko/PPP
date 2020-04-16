#include <CL/sycl.hpp>
//#include "Util.hpp"
#include <stdlib.h>
#include <iostream>
#include <complex>
using namespace cl::sycl;
int main(int argc, char** argv) {
  //process_args(argc, argv);
  //init();
  int n = 3;
  auto q  = queue(gpu_selector{});

  std::complex<float>* t = (std::complex<float>*)malloc_shared(n * sizeof(std::complex<float>), q); 
  auto nn = n;
  q.submit([&] (handler& cgh) { // q scope
    cgh.parallel_for<class oneplusone>(range<1>(n), [=](id<1> i) {
      for (int j = 0; j < nn; j++)
        t[i] = t[0] * t[j];
    }); // end task scope
  }); // end q scope
  q.wait();
  std::cout << t[0] << std::endl;

#pragma omp target map(tofrom:t[0:3])
  {
    for (int j = 0; j < nn; j++)
      t[0] = t[0] * t[j];
  }

  return 0;
}
