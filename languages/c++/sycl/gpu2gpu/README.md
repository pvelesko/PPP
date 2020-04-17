# Using SYCL with multiple Devices
This test shows how to use SYCL with multiple offload devices.
I ran this test on dgx2 system which has 8 V100 GPUs

## Tests
#### Test 0
Allocate on the host with malloc_host, access from all devices. Expected to work
#### Test 1
Allocate on the host with malloc_shared, access from all devices. Expected to work
#### Test 2
Allocate on the device with malloc_device, allocate a host pointer tied to GPU context. 
Need this to copy GPU data to the host since malloc_device allocations are not
copyable to CPU directly. 
#### Test 3
Allocate on the device with malloc_device, try to access from another device. Might or might not
be supported by SYCL implementation

## Compile and Run
You will need latest intel branch of llvm that supports SYCL offload on Nvidia GPUs

````
make cuda
./cuda
````


