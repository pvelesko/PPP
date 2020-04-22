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

#include <iostream>

#include "GSimulation.hpp"

int main(int argc, char** argv) 
{
  int N;			//number of particles
  int nstep; 		//number ot integration steps
  
  GSimulation sim;
   std::string a ;
  if(argc>1)
  {
    N=atoi(argv[1]);
    sim.set_number_of_particles(N);  
    if(argc>2) 
    {
      nstep=atoi(argv[2]);
      sim.set_number_of_steps(nstep);  
    }
    if(argc > 3) 
      a = argv[3];
      std::cout << a << std::endl;
      if (!a.compare("cpu")) sim.set_devices(1);
      if (!a.compare(std::string("gpu"))) sim.set_devices(2);
      if (!a.compare(std::string("cpu+gpu"))) sim.set_devices(3);

    if(argc > 4) 
      sim.set_cpu_ratio(atof(argv[4]));

    if(argc > 5) {
      sim.set_cpu_wgsize(atoi(argv[5]));
      sim.set_gpu_wgsize(atoi(argv[6]));
    }
  }
  
  sim.start();

  return 0;
}
