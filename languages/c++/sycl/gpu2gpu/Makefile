.PHONY: clean all
CFLAGS=-O2 -std=c++14 -g

all: intel

intel: main.cpp SyclUtil.hpp
	onecxx -fsycl -I./ ./main.cpp -o intel ${CFLAGS}

codeplay: main.cpp SyclUtil.hpp
	compute++ -sycl -sycl-driver -lComputeCpp ./main.cpp -o intel ${CFLAGS}

hipsycl: main.cpp SyclUtil.hpp
	syclcc-clang-wrapper ./main.cpp -o hipsycl ${CFLAGS}

cuda: main.cpp SyclUtil.hpp
	clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice ./main.cpp -o cuda ${CFLAGS}

clean:
	rm -f ./intel ./codeplay ./hipsycl ./cuda
