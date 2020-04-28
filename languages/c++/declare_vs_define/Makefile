.PHONY: clean all
CFLAGS=-O2 -std=c++14 -g

all: intel

intel: main.cpp Util.hpp
	icpc -I./ ./main.cpp -o intel ${CFLAGS}

clang: main.cpp Util.hpp
	clang++ ./main.cpp -o llvmclang ${CFLAGS}

gcc: main.cpp Util.hpp
	g++ ./main.cpp -o gnugcc ${CFLAGS}

clean:
	rm -f ./intel ./llvmclang ./gnugcc
