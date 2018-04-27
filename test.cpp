#include <iostream>
#include <cstdint>

int main(int argc, char** argv) {
  std::uint64_t a;
  std::uint64_t b;
  a = a - 2;
  b = 2;
  a = a + b;
  std::cout << a << "\n";
}
