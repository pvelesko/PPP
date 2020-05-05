.PHONY: clean all

all: intel

KOKKOS_HOME=./build_velesko/install
intel: main.cpp
	$(CC) ./main.cpp -o intel -I${KOKKOS_HOME}/include -L${KOKKOS_HOME}/lib -lkokkos -g

clean:
	rm -f ./intel ./codeplay codeplay_ptx
