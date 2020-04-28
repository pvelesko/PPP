#include <stdlib.h>
#include <iostream>
int n;
inline void process_args(int argc, char** argv) {
  if (argc > 1) {
    n = std::atoi(argv[1]);
  } else {
    n = 3;
  }
  std::cout << "Using N = " << n << std::endl;
}

template<class T>
inline void dump(const std::string name, T* var) {
  for(int i = 0; i < n; i++)
    std::cout << name << "[" << i << "] = " << var[i] << std::endl;
}
template<class T>
inline void dump(const std::string name, T* var, int size) {
  for(int i = 0; i < size; i++)
    std::cout << name << "[" << i << "] = " << var[i] << std::endl;
}

template<class T>
inline void dump(const std::string name, T* var, int start, int end) {
  for(int i = start; i < end; i++)
    std::cout << name << "[" << i << "] = " << var[i] << std::endl;
}
