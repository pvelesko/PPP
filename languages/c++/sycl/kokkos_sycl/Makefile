.PHONY: clean all

CXXFLAGS=-qnextgen -g -O2 -fsycl -fsycl-unnamed-lambda -std=c++17 --gcc-toolchain=/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/gcc-9.3.0-qfmcwfbuvnpn47zxjzfjvodzjl6reerh/ -ldl
#CXXFLAGS=-qnextgen -g -O2 -fiopenmp -std=c++17 --gcc-toolchain=/soft/packaging/spack-builds/linux-rhel7-x86_64/gcc-9.3.0/gcc-9.3.0-qfmcwfbuvnpn47zxjzfjvodzjl6reerh/ -ldl
KOKKOS=-I${KOKKOS_HOME}/include -L${KOKKOS_HOME}/lib -lkokkos

all: intel

main.o: main.cpp
	icpc $(CXXFLAGS) -c ./main.cpp $(KOKKOS)
intel: main.o
	icpc $(CXXFLAGS) ./main.o -o intel $(KOKKOS) 
clean:
	rm -f ./intel ./codeplay codeplay_ptx *.o
