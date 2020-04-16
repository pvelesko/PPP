#include <CL/sycl.hpp>
#include "SyclUtil.hpp"
#include <stdlib.h>
#include <iostream>
using namespace cl::sycl;
int main(int argc, char** argv) {
  process_args(argc, argv);
  init();
  return 0;
}
