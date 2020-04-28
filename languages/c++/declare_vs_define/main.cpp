#include "Util.hpp"
class Member {
  public:
  Member() = delete;
  Member(int n) {
    std::cout << "Called Member Default Constructor" << std::endl;
  }

};

class Test {
  private:
  Member mem;

  public:
  //Test() {}; // Won't work because will implicitly call Member() which is deleted
  //Test() { // Won't work because will call Member() before calling Member(int n)
  //  mem(0);
  //}; 
  Test() : mem(0) {}; //OK

};

int main(int argc, char** argv) {
  process_args(argc, argv);
  Test test();
  return 0;
}
