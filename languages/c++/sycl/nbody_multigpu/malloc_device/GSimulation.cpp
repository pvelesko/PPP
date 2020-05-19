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
  event* e = new event[num_devices];

  /* malloc_host allows all the GPUs in gpu_context to access this data */
  particles = (ParticleSoA*)      malloc(sizeof(ParticleSoA));
  particles->pos_x = (real_type*) malloc(n*sizeof(real_type));
  particles->pos_y = (real_type*) malloc(n*sizeof(real_type));
  particles->pos_z = (real_type*) malloc(n*sizeof(real_type));
  particles->vel_x = (real_type*) malloc(n*sizeof(real_type));
  particles->vel_y = (real_type*) malloc(n*sizeof(real_type));
  particles->vel_z = (real_type*) malloc(n*sizeof(real_type));
  particles->acc_x = (real_type*) malloc(n*sizeof(real_type));
  particles->acc_y = (real_type*) malloc(n*sizeof(real_type));
  particles->acc_z = (real_type*) malloc(n*sizeof(real_type));
  particles->mass  = (real_type*) malloc(n*sizeof(real_type));

  std::vector<float*> d_particles_acc_x;
  std::vector<float*> d_particles_acc_y;
  std::vector<float*> d_particles_acc_z;

  std::vector<float*> d_particles_pos_x;
  std::vector<float*> d_particles_pos_y;
  std::vector<float*> d_particles_pos_z;

  std::vector<float*> d_particles_mass;
  for(int i = 0; i < num_devices; i++) {
    d_particles_pos_x.push_back((real_type*) malloc_device(n*sizeof(real_type), q[i]));
    d_particles_pos_y.push_back((real_type*) malloc_device(n*sizeof(real_type), q[i]));
    d_particles_pos_z.push_back((real_type*) malloc_device(n*sizeof(real_type), q[i]));
    d_particles_acc_x.push_back((real_type*) malloc_device(n*sizeof(real_type), q[i]));
    d_particles_acc_y.push_back((real_type*) malloc_device(n*sizeof(real_type), q[i]));
    d_particles_acc_z.push_back((real_type*) malloc_device(n*sizeof(real_type), q[i]));
    d_particles_mass .push_back((real_type*) malloc_device(n*sizeof(real_type), q[i]));
  }

  init_pos();	
  init_vel();
  init_acc();
  init_mass();

  for(int i = 0; i < num_devices; i++) {
    e[i] = q[i].memcpy(d_particles_pos_x[i], particles->pos_x, n * sizeof(real_type));
    e[i] = q[i].memcpy(d_particles_pos_y[i], particles->pos_y, n * sizeof(real_type));
    e[i] = q[i].memcpy(d_particles_pos_z[i], particles->pos_z, n * sizeof(real_type));

    e[i] = q[i].memcpy(d_particles_acc_x[i], particles->acc_x, n * sizeof(real_type));
    e[i] = q[i].memcpy(d_particles_acc_y[i], particles->acc_y, n * sizeof(real_type));
    e[i] = q[i].memcpy(d_particles_acc_z[i], particles->acc_z, n * sizeof(real_type));

    e[i] = q[i].memcpy(d_particles_mass[i],  particles->mass,  n * sizeof(real_type));
  }
  
     // wait for all GPUs to complete
    for (int qi = 0; qi < num_devices; qi++)
      q[qi].wait();
 
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



  // Set up Max Total Threads
  auto num_groups = q[0].get_device().get_info<info::device::max_compute_units>();
  auto work_group_size =q[0].get_device().get_info<info::device::max_work_group_size>();
  auto total_threads = (int)(num_groups * work_group_size);

  const double t0 = time.start();
  for (int s=1; s<=get_nsteps(); ++s)
  { // time step loop
    ts0 += time.start(); 

    for (int qi = 0; qi < q.size(); qi++) {
    auto particles_acc_x = d_particles_acc_x[qi];
    auto particles_acc_y = d_particles_acc_y[qi];
    auto particles_acc_z = d_particles_acc_z[qi];

    auto particles_pos_x = d_particles_pos_x[qi];
    auto particles_pos_y = d_particles_pos_y[qi];
    auto particles_pos_z = d_particles_pos_z[qi];

    auto particles_mass  = d_particles_mass[qi];

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
      real_type ax_i = 0;
      real_type ay_i = 0;
      real_type az_i = 0;
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

    // copy from devices
  for(int i = 0; i < num_devices; i++) {
    e[i] = q[i].memcpy(particles->acc_x + starts[i],d_particles_acc_x[i] + starts[i], (ends[i]-starts[i]) * sizeof(real_type));
    e[i] = q[i].memcpy(particles->acc_y + starts[i],d_particles_acc_y[i] + starts[i], (ends[i]-starts[i]) * sizeof(real_type));
    e[i] = q[i].memcpy(particles->acc_z + starts[i],d_particles_acc_z[i] + starts[i], (ends[i]-starts[i]) * sizeof(real_type));
  }

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

    // copy to devices
  for(int i = 0; i < num_devices; i++) {
    e[i] = q[i].memcpy(d_particles_pos_x[i], particles->pos_x, n * sizeof(real_type));
    e[i] = q[i].memcpy(d_particles_pos_y[i], particles->pos_y, n * sizeof(real_type));
    e[i] = q[i].memcpy(d_particles_pos_z[i], particles->pos_z, n * sizeof(real_type));
  }
    for (int qi = 0; qi < num_devices; qi++)
      e[qi].wait();

     
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
 free(particles->pos_x);
 free(particles->pos_y);
 free(particles->pos_z);
 free(particles->vel_x);
 free(particles->vel_y);
 free(particles->vel_z);
 free(particles->acc_x);
 free(particles->acc_y);
 free(particles->acc_z);
 free(particles->mass);
 free(particles);
}
