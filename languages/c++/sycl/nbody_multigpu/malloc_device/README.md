# Nbody Simulation with SYCL Multi-GPU Support

## Implementation
Multiple GPUs are exploited by creating a SYCL queue for each gpu device.
Each GPU gets N/num_gpus of outer-loop work. 
Submit work to all GPUs, store events
Wait events to finish
Reduce on the host. 

## Performance
Single V100 gets ~10000 GFLOPs Single Precision. This version scales poorly with the number of GPUs.

## Future improvements
1. Create malloc_device memory for each GPU and use deep copies instead of relying on malloc_host arrays
2. Reduce on the device

## Compile
Requires intel branch of llvm with SYCL support for CUDA
 -DMAXTHREADS flag changes how work is scheduled. If provided, maximum number of GPU threads will be spawned. Improves performance on GPUs

## Run & Check Correctness
./nbody.x 2000 500

Expected result for this input kenergy = 571.53
===============================
 Initialize Gravity Simulation
 nPart = 2000; nSteps = 500; dt = 0.1
 ------------------------------------------------
 s       dt      kenergy     time (s)    GFlops
 ------------------------------------------------
 50      5       0.1432      0.26196     22.148
 100     10      2.4341      0.19766     29.353
 150     15      8.1256      0.19748     29.38
 200     20      17.877      0.19733     29.403
 250     25      32.966      0.20042     28.948
 300     30      55.786      0.19829     29.26
 350     35      91.132      0.19331     30.013
 400     40      150.12      0.19377     29.942
 450     45      264.78      0.19502     29.75
 500     50      571.53      0.20833     27.85

