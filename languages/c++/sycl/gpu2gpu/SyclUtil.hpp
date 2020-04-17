using namespace cl::sycl;
int n;
queue q;
device dev;
context ctx;

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


inline void init() {
  std::string env;
  if (std::getenv("SYCL_DEVICE") != NULL) {
    env = std::string(std::getenv("SYCL_DEVICE"));
  } else {
    env = std::string("");
  }
  std::cout << "Using DEVICE = " << env << std::endl;
  if (!env.compare("gpu") or !env.compare("GPU")) {
    q = cl::sycl::queue(cl::sycl::gpu_selector{}, exception_handler);
  } else if (!env.compare("cpu") or !env.compare("CPU")) {
    q = cl::sycl::queue(cl::sycl::cpu_selector{}, exception_handler);
  } else if (!env.compare("host") or !env.compare("HOST")) {
    q = cl::sycl::queue(cl::sycl::host_selector{}, exception_handler);
  } else {
    q = cl::sycl::queue(cl::sycl::default_selector{}, exception_handler);
  }
  dev = q.get_device();
  ctx = q.get_context();
  std::cout << "Running on "
            << dev.get_info<info::device::name>()
            << std::endl;
};

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


