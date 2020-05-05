/*
    This file is part of the example codes which have been used
    for the "Code Optmization Workshop".
    
    Copyright (C) 2016  Fabio Baruffa <fbaru-dev@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <CL/sycl.hpp>
#include <random>

#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <stdlib.h>


#include "GSimulation.hpp"
#include "cpu_time.hpp"

template<class T>
inline void dump(const std::string name, T* var, int start, int end) {
  for(int i = start; i < end; i++)
    std::cout << name << "[" << i << "] = " << var[i] << std::endl;
}


auto exception_handler = [] (cl::sycl::exception_list exceptions) {
  for (std::exception_ptr const& e : exceptions) {
    try {
  std::rethrow_exception(e);
    } catch(cl::sycl::exception const& e) {
  std::cout << "Caught asynchronous SYCL exception:\n"
        << e.what() << std::endl;
    }
  }
};

class CUDASelector : public cl::sycl::device_selector {
  public:
    int operator()(const cl::sycl::device &Device) const override {
      using namespace cl::sycl::info;

      const std::string DriverVersion = Device.get_info<device::driver_version>();

      if (Device.is_gpu() && (DriverVersion.find("CUDA") != std::string::npos)) {
        return 1;
      };
      return -1;
    }
};

using namespace cl::sycl;
context gpu_context;

GSimulation :: GSimulation()
{
  std::cout << "===============================" << std::endl;
  std::cout << " Initialize Gravity Simulation" << std::endl;
  set_npart(2000); 
  set_nsteps(500);
  set_tstep(0.1); 
  set_sfreq(50);
}

void GSimulation :: set_number_of_particles(int N)  
{
  set_npart(N);
}

void GSimulation :: set_number_of_steps(int N)  
{
  set_nsteps(N);
}

void GSimulation :: init_pos()
{
  std::random_device rd;        //random number generator
  std::mt19937 gen(42);
  std::uniform_real_distribution<real_type> unif_d(0,1.0);

  for(int i=0; i<get_npart(); ++i)
  {
    particles->pos_x[i] = unif_d(gen);
    particles->pos_y[i] = unif_d(gen);
    particles->pos_z[i] = unif_d(gen);
  }
}

void GSimulation :: init_vel()
{
  std::random_device rd;        //random number generator
  std::mt19937 gen(42);
  std::uniform_real_distribution<real_type> unif_d(-1.0,1.0);

  for(int i=0; i<get_npart(); ++i)
  {
    particles->vel_x[i] = unif_d(gen) * 1.0e-3f;
    particles->vel_y[i] = unif_d(gen) * 1.0e-3f;
    particles->vel_z[i] = unif_d(gen) * 1.0e-3f;
  }
}

void GSimulation :: init_acc()
{
  for(int i=0; i<get_npart(); ++i)
  {
    particles->acc_x[i] = 0.f;
    particles->acc_y[i] = 0.f;
    particles->acc_z[i] = 0.f;
  }
}

void GSimulation :: init_mass()
{
  real_type n   = static_cast<real_type> (get_npart());
  std::random_device rd;        //random number generator
  std::mt19937 gen(42);
  std::uniform_real_distribution<real_type> unif_d(0.0,1.0);

  for(int i=0; i<get_npart(); ++i)
  {
    particles->mass[i] = n * unif_d(gen);
  }
}

void GSimulation :: start() 
{

  auto gpu_platform = platform(gpu_selector{});
  auto gpu_devices = gpu_platform.get_devices(); 
  gpu_context = context(gpu_platform);

  std::cout << "Number of GPU Devices Available: " << gpu_devices.size() << std::endl;
  std::vector<queue> q{}; // queue vector 
  int use_gpus;
  if(get_number_of_gpus() != 0) {
    use_gpus = get_number_of_gpus();
  } else {
    use_gpus = gpu_devices.size();
  }
  std::cout << "Number of GPU Devices to Use: " << use_gpus << std::endl;

  for (int i = 0; i < use_gpus; i++)
    q.push_back(queue(gpu_devices[i], exception_handler));

  std::cout << "Number of queues available: " << q.size() << std::endl;

  int num_devices = q.size();
  int n = get_npart();

  // data/array offsets for splitting work between GPUs
  int* starts = new int(num_devices);
  int* ends = new int(num_devices);
  int share = n / num_devices;
  int remainder = n % num_devices;

  /* Setup arrays for start/end indices for each GPU */
  starts[0] = 0; ends[0] = share;
  for (int i = 1; i < num_devices; i++) {
    starts[i] = ends[i - 1];
    ends[i] = starts[i] + share;
  }
  ends[num_devices - 1] += remainder;

  // print device names
  for (int i = 0; i < q.size(); i++) {
    std::cout << "Device #" << i << ": ";
    std::cout << q[i].get_device().get_info<info::device::name>() << std::endl;
  }

  real_type energy;
  real_type dt = get_tstep();
  int i;

  /* malloc_host allows all the GPUs in gpu_context to access this data */
  particles = (ParticleSoA*) malloc_host(sizeof(ParticleSoA),      gpu_context);
  particles->pos_x = (real_type*) malloc_host(n*sizeof(real_type), gpu_context);
  particles->pos_y = (real_type*) malloc_host(n*sizeof(real_type), gpu_context);
  particles->pos_z = (real_type*) malloc_host(n*sizeof(real_type), gpu_context);
  particles->vel_x = (real_type*) malloc_host(n*sizeof(real_type), gpu_context);
  particles->vel_y = (real_type*) malloc_host(n*sizeof(real_type), gpu_context);
  particles->vel_z = (real_type*) malloc_host(n*sizeof(real_type), gpu_context);
  particles->acc_x = (real_type*) malloc_host(n*sizeof(real_type), gpu_context);
  particles->acc_y = (real_type*) malloc_host(n*sizeof(real_type), gpu_context);
  particles->acc_z = (real_type*) malloc_host(n*sizeof(real_type), gpu_context);
  particles->mass  = (real_type*) malloc_host(n*sizeof(real_type), gpu_context);

  init_pos();	
  init_vel();
  init_acc();
  init_mass();
  
  print_header();
  
  _totTime = 0.; 
 
  const float softeningSquared = 1.e-3f;
  const float G = 6.67259e-11f;
  
  CPUTime time;
  double ts0 = 0;
  double ts1 = 0;
  double nd = double(n);
  double gflops = 1e-9 * ( (11. + 18. ) * nd*nd  +  nd * 19. );
  double av=0.0, dev=0.0;
  int nf = 0;


  auto* particles_acc_x = particles->acc_x;
  auto* particles_acc_y = particles->acc_y;
  auto* particles_acc_z = particles->acc_z;

  auto* particles_pos_x = particles->pos_x;
  auto* particles_pos_y = particles->pos_y;
  auto* particles_pos_z = particles->pos_z;

  auto* particles_mass = particles->mass;

  // Set up Max Total Threads
  auto num_groups = q[0].get_device().get_info<info::device::max_compute_units>();
  auto work_group_size =q[0].get_device().get_info<info::device::max_work_group_size>();
  auto total_threads = (int)(num_groups * work_group_size);

  event* e = new event[num_devices];
  const double t0 = time.start();
  for (int s=1; s<=get_nsteps(); ++s)
  { // time step loop
    ts0 += time.start(); 

    for (int qi = 0; qi < q.size(); qi++) {
    int start, end;
    start = starts[qi];
    end = ends[qi];
    auto r = range<1>(end-start);
    e[qi] = q[qi].submit([&] (handler& cgh) {

       cgh.parallel_for<class update_accel>(
#ifdef MAXTHREADS
       range<1>(total_threads), [=](item<1> item)
#else
       range<1>(end - start), [=](item<1> item)
#endif

       { // lambda start
#ifdef MAXTHREADS
      for (int i = item.get_id()[0] + start; i < end; i+=total_threads)
#else
      auto i = item.get_id()[0] + start;
#endif
      { 
      real_type ax_i = particles_acc_x[i];
      real_type ay_i = particles_acc_y[i];
      real_type az_i = particles_acc_z[i];
      for (int j = 0; j < n; j++) {
        real_type dx, dy, dz;
     	  real_type distanceSqr = 0.0f;
     	  real_type distanceInv = 0.0f;
     	     
     	  dx = particles_pos_x[j] - particles_pos_x[i];	//1flop
     	  dy = particles_pos_y[j] - particles_pos_y[i];	//1flop	
     	  dz = particles_pos_z[j] - particles_pos_z[i];	//1flop
     	
        distanceSqr = dx*dx + dy*dy + dz*dz + softeningSquared;	//6flops
        distanceInv = 1.0f / sqrt(distanceSqr);			//1div+1sqrt
     
     	  ax_i += dx * G * particles_mass[j] * distanceInv * distanceInv * distanceInv; //6flops
     	  ay_i += dy * G * particles_mass[j] * distanceInv * distanceInv * distanceInv; //6flops
     	  az_i += dz * G * particles_mass[j] * distanceInv * distanceInv * distanceInv; //6flops
      }
      particles_acc_x[i] = ax_i;
      particles_acc_y[i] = ay_i;
      particles_acc_z[i] = az_i;
      } // end of max_threads loop or just scope
     }); // parallel_for
   }); // queue scope
    
    } // queue loop

    // wait for all GPUs to complete
    for (int qi = 0; qi < num_devices; qi++)
      e[qi].wait();

    // reduce on the host.. this could be done on GPU
    energy = 0;
    for (int i = 0; i < n; ++i)// update position
    {

      particles->vel_x[i] += particles->acc_x[i] * dt; //2flops
      particles->vel_y[i] += particles->acc_y[i] * dt; //2flops
      particles->vel_z[i] += particles->acc_z[i] * dt; //2flops
	   
      particles->pos_x[i] += particles->vel_x[i] * dt; //2flops
      particles->pos_y[i] += particles->vel_y[i] * dt; //2flops
      particles->pos_z[i] += particles->vel_z[i] * dt; //2flops


          particles->acc_x[i] = 0.;
          particles->acc_y[i] = 0.;
          particles->acc_z[i] = 0.;

      energy += particles->mass[i] * (
	        particles->vel_x[i]*particles->vel_x[i] + 
                particles->vel_y[i]*particles->vel_y[i] +
                particles->vel_z[i]*particles->vel_z[i]); //7flops
    }
   _kenergy = 0.5 * energy; 

     
     ts1 += time.stop();
     if(!(s%get_sfreq()) ) 
     {
       nf += 1;      
       std::cout << " " 
	 	<<  std::left << std::setw(8)  << s
	 	<<  std::left << std::setprecision(5) << std::setw(8)  << s*get_tstep()
	 	<<  std::left << std::setprecision(5) << std::setw(12) << _kenergy
	 	<<  std::left << std::setprecision(5) << std::setw(12) << (ts1 - ts0)
	 	<<  std::left << std::setprecision(5) << std::setw(12) << gflops*get_sfreq()/(ts1 - ts0)
	 	<<  std::endl;
       if(nf > 2) 
       {
	 av  += gflops*get_sfreq()/(ts1 - ts0);
	 dev += gflops*get_sfreq()*gflops*get_sfreq()/((ts1-ts0)*(ts1-ts0));
       }
       
       ts0 = 0;
       ts1 = 0;
     }
  } //end of the time step loop

  const double t1 = time.stop();
  _totTime  = (t1-t0);
  _totFlops = gflops*get_nsteps();
  
  av/=(double)(nf-2);
  dev=sqrt(dev/(double)(nf-2)-av*av);
  
  int nthreads=1;

  std::cout << std::endl;
  std::cout << "# Number Threads     : " << nthreads << std::endl;	   
  std::cout << "# Total Time (s)     : " << _totTime << std::endl;
  std::cout << "# Average Perfomance : " << av << " +- " <<  dev << std::endl;
  std::cout << "===============================" << std::endl;

}


void GSimulation :: print_header()
{
	    
  std::cout << " nPart = " << get_npart()  << "; " 
	    << "nSteps = " << get_nsteps() << "; " 
	    << "dt = "     << get_tstep()  << std::endl;
	    
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << " " 
	    <<  std::left << std::setw(8)  << "s"
	    <<  std::left << std::setw(8)  << "dt"
	    <<  std::left << std::setw(12) << "kenergy"
	    <<  std::left << std::setw(12) << "time (s)"
	    <<  std::left << std::setw(12) << "GFlops"
	    <<  std::endl;
  std::cout << "------------------------------------------------" << std::endl;


}

GSimulation :: ~GSimulation()
{
 free(particles->pos_x, gpu_context);
 free(particles->pos_y, gpu_context);
 free(particles->pos_z, gpu_context);
 free(particles->vel_x, gpu_context);
 free(particles->vel_y, gpu_context);
 free(particles->vel_z, gpu_context);
 free(particles->acc_x, gpu_context);
 free(particles->acc_y, gpu_context);
 free(particles->acc_z, gpu_context);
 free(particles->mass,  gpu_context);
 free(particles,        gpu_context);
}
