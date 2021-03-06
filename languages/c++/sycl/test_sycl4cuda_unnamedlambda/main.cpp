#include <CL/sycl.hpp>
#include "SyclUtil.hpp"
#include <stdlib.h>
#include <iostream>
using namespace cl::sycl;
int main(int argc, char** argv) {
  process_args(argc, argv);
  init();

  int* num = (int*) malloc_shared(3 * sizeof(int), q);
  q.submit([&](handler& cgh) {
    cgh.single_task([=]() {
      num[0] = 0;
    }); // task
  }); // queue

  return 0;
}
