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
  int n = get_npart();
  std::cout << "device option: " << get_devices() << std::endl;
  std::vector<queue> q;
  int num_devices = 0;
  if (get_devices() == 1) {
    q.push_back(queue(cpu_selector()));
    num_devices = 1;
  } else if (get_devices() == 2) {
    q.push_back(queue(gpu_selector()));
    num_devices = 1;
  } else {
    num_devices = 2;
    q.push_back(queue(cpu_selector()));
    q.push_back(queue(gpu_selector()));
  }

    /* Set up workgroup sizes
     * very naive implementaiton assuming device[0] is CPU
     * and device[1] is GPU. If using 1 device, say only GPU
     * which is normally the 2nd platform that's found it's
     * best to use the same wgsize arguments:
     * ./nbody.x 2000 5000 -1 256 256
     * -1 for no tuning of work-split
     *  256 local size for all devices. Currently max of 2
     */
    if (_cpu_wgsize != 0 and _gpu_wgsize != 0) {
      //local[0] = cl::NDRange(_cpu_wgsize);
      //local[1] = cl::NDRange(_gpu_wgsize);
      printf("CPU WorkGroup Size:%d\n", _cpu_wgsize);
      printf("GPU WorkGroup Size:%d\n", _gpu_wgsize);
    } else {
      printf("Using automatic WorkGroup sizes\n");
    }

    /* Set work ratio between CPU/GPU
     * if no arg is passed it will test all
     * if -1 is passed it will test all
     * else, use the provided ratio
     */
    float cpu_ratio;
    bool tuning;
    if (num_devices > 1) {
      cpu_ratio = _cpu_ratio;
      tuning = (cpu_ratio < 0);
      if (tuning) cpu_ratio = 0.01; // not 0 because can't create 0 length buffers
    } else {
      cpu_ratio = 1.0f;
      tuning = false;
    }


  // data/array offsets for splitting work CPU/GPU
  int* shares = new int(num_devices);
  int* offsets = new int(num_devices);


  std::cout << "CPU to GPU work ratio: " << cpu_ratio << std::endl;
  //
  // print device names
  for (int i = 0; i < q.size(); i++) {
    std::cout << "Device #" << i << ": ";
    std::cout << q[i].get_device().get_info<info::device::name>() << std::endl;
  }


  real_type energy;
  real_type dt = get_tstep();
  int i;

  const int alignment = 32;
  device d = q[0].get_device();
  context ctx = q[0].get_context();
  particles = static_cast<ParticleSoA*>(malloc_shared(sizeof(ParticleSoA), d, ctx));

  particles->pos_x = static_cast<real_type*>(malloc_shared(n*sizeof(real_type), d, ctx));
  particles->pos_y = static_cast<real_type*>(malloc_shared(n*sizeof(real_type), d, ctx));
  particles->pos_z = static_cast<real_type*>(malloc_shared(n*sizeof(real_type), d, ctx));
  particles->vel_x = static_cast<real_type*>(malloc_shared(n*sizeof(real_type), d, ctx));
  particles->vel_y = static_cast<real_type*>(malloc_shared(n*sizeof(real_type), d, ctx));
  particles->vel_z = static_cast<real_type*>(malloc_shared(n*sizeof(real_type), d, ctx));
  particles->acc_x = static_cast<real_type*>(malloc_shared(n*sizeof(real_type), d, ctx));
  particles->acc_y = static_cast<real_type*>(malloc_shared(n*sizeof(real_type), d, ctx));
  particles->acc_z = static_cast<real_type*>(malloc_shared(n*sizeof(real_type), d, ctx));
  particles->mass  = static_cast<real_type*>(malloc_shared(n*sizeof(real_type), d, ctx));

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
  
  auto tmp = particles; // temp fix due to bug 
  const double t0 = time.start();
  for (int s=1; s<=get_nsteps(); ++s)
  { // time step loop
    ts0 += time.start(); 
    shares[0] = n * cpu_ratio;
    offsets[0] = 0;
    for (int i = 1; i < num_devices; i++) {
      shares[i] = n - n * cpu_ratio;
      offsets[i] = n * cpu_ratio;
    }

    //for (int qi = 0; qi < q.size(); qi++)
      int qi = 0;
      auto e = q[qi].submit([&] (handler& cgh)  {
        cgh.parallel_for<class update_accel>(
          nd_range<1>(shares[qi], 0, 0), [=](nd_item<1> item) {

            int i = item.get_global_id()[0];
            real_type ax_i = tmp->acc_x[i];
            real_type ay_i = tmp->acc_y[i];
            real_type az_i = tmp->acc_z[i];

            for (int j = 0; j < n; j++)
            {
              real_type dx, dy, dz;
	            real_type distanceSqr = 0.0f;
	            real_type distanceInv = 0.0f;
	               
	            dx = tmp->pos_x[j] - tmp->pos_x[i];	//1flop
	            dy = tmp->pos_y[j] - tmp->pos_y[i];	//1flop	
	            dz = tmp->pos_z[j] - tmp->pos_z[i];	//1flop
 
 	            distanceSqr = dx*dx + dy*dy + dz*dz + softeningSquared;	//6flops
 	            distanceInv = 1.0f / sqrt(distanceSqr);			//1div+1sqrt

	            ax_i += dx * G * tmp->mass[j] * distanceInv * distanceInv * distanceInv; //6flops
	            ay_i += dy * G * tmp->mass[j] * distanceInv * distanceInv * distanceInv; //6flops
	            az_i += dz * G * tmp->mass[j] * distanceInv * distanceInv * distanceInv; //6flops
            }
            tmp->acc_x[i] = ax_i;
            tmp->acc_y[i] = ay_i;
            tmp->acc_z[i] = az_i;

        }); // end of parallel for scope
      }); // end of command group scope

    e.wait();
    // no device side reductions so have to do it here
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
       if (tuning)
        {
          printf("CPU/GPU ratio = %f\n", cpu_ratio);
          cpu_ratio += 0.01;
          if (cpu_ratio > 1.0f) cpu_ratio = 1.0f;
        }

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
//  free(particles->pos_x);
//  free(particles->pos_y);
//  free(particles->pos_z);
//  free(particles->vel_x);
//  free(particles->vel_y);
//  free(particles->vel_z);
//  free(particles->acc_x);
//  free(particles->acc_y);
//  free(particles->acc_z);
//  free(particles->mass);
//  free(particles);

}
