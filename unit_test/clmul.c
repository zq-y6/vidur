#include <stdio.h>
#include <stdint.h>

uint64_t clmul64(uint64_t x, uint64_t y) {
  uint64_t result = 0;
  for (int i = 0; i < 64; ++i) {
    if (y & (1ULL << i))
      result ^= (x << i);
  }
  return result;
}

int main() {
  /* 0b1011 * 0b1101 = 1011000 ^ 101100 ^ 1011 = 1111111 */
  printf("%lx clmul %lx is %lx\n", (uint64_t)0xb, (uint64_t)0xd, clmul64((uint64_t)0xb, (uint64_t)0xd));
}