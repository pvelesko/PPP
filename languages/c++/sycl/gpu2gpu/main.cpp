#include <CL/sycl.hpp>
#include "SyclUtil.hpp"
#include <stdlib.h>
#include <iostream>
#include <tuple>
#include <assert.h>
using namespace cl::sycl;
int main(int argc, char** argv) {
  process_args(argc, argv);
  //init();

  auto gpu_platform = platform(gpu_selector{});
  auto gpu_devices = gpu_platform.get_devices(); 
  auto gpu_context = context(gpu_platform);

  std::cout << "Number of GPU Devices: " << gpu_devices.size() << std::endl;
  std::vector<queue> qs{}; // queue vector 
  for (auto& dev : gpu_devices)
    qs.push_back(queue(dev, exception_handler));
  std::cout << "Number of queues available: " << qs.size() << std::endl;
  if (qs.size() < 2) {
    std::cout << "This test requires multiple GPUs, like a DGX system" << std::endl;
    exit(0);
  }

  
  { /* test 0 
  malloc_host creates memory on the host that can be accessed by devices
  in the same context as what was given to malloc_hot */
  std::cout << "Test 0, malloc_host, access from all GPUs" << std::endl;
  int* p = static_cast<int*>(malloc_host(1 * sizeof(int), gpu_context));
  for (int qi = 0; qi < qs.size(); qi++) {
    qs[qi].submit([&](handler& cgh) {
      cgh.single_task<class t0>([=]() {
        p[0]++;
      }); // task
    }); //queue
  } // end test 0
  assert(p[0] == qs.size());
  } /* end test 0 */

  { /* test 1 create shared allocation
       Shared allocations can be allocated on host or a specific device
       May allow device-to-device communication
  */
  std::cout << "Test 1, malloc_shared, access from CPU and GPUs" << std::endl;
  int* p = static_cast<int*>(malloc_shared(1 * sizeof(int), qs[0]));
  event e;
  p[0] = 0; // test access from host
  assert(p[0] == 0);
  for (int qi = 0; qi < qs.size(); qi++) {
    e = qs[qi].submit([&](handler& cgh) { // test access from another device
      cgh.single_task<class t1>([=]() {
        p[0]++;
      }); // task
    }); //queue
    e.wait();
  }
  assert(p[0] == qs.size());
  } /* end test 1 */

  { /* test 2 create device allocation
       Device allocations allocate directly on the device
       May allow device-to-device communication
  */
  std::cout << "Test 2, malloc_device on each device" << std::endl;
  std::vector<int*> ptrs{qs.size()};
  int* p = static_cast<int*>(malloc_host(1 * sizeof(int), gpu_context));
  for (int qi = 0; qi < qs.size(); qi++) {
    ptrs[qi] = static_cast<int*>(malloc_device(1 * sizeof(int), qs[qi]));
  }

  event e;
  for (int qi = 0; qi < qs.size(); qi++) {
    int* dat = ptrs[qi]; // need to do this to avoid passing vector into kernel
    // vector is not trivially copyable and will not work
    e = qs[qi].submit([&](handler& cgh) { // test access from another device
      cgh.single_task<class t2>([=]() {
        dat[0]++;
        p[0] += dat[0]; // write to something that host can later access
      }); // task
    }); //queue
    e.wait();
  }
  
  assert(p[0] == qs.size());
  } /* end test 2 */

  { /* test 3 create device allocation
       Device allocations allocate directly on the device
       May allow device-to-device communication
  */
  std::cout << "Test 3, malloc_device on one gpu, access from other devices" << std::endl;
  int* p = static_cast<int*>(malloc_device(1 * sizeof(int), qs[0]));
  event e;
  for (int qi = 0; qi < qs.size(); qi++) {
    e = qs[qi].submit([&](handler& cgh) { // test access from another device
      cgh.single_task<class t3>([=]() {
        p[0]++;
      }); // task
    }); //queue
    e.wait();
  }
  //assert(p[0] == qs.size());
  } /* end test 3 */


  return 0;
}
