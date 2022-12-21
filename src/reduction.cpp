#include <stdint.h>
#include <cstddef>

#include "reduction.hpp"

using namespace std;

uint32_t reduce(const uint32_t *data, size_t n) {
    uint32_t ret = 0.0;
    for (size_t i = 0; i < n; ++i) {
        ret += data[i];
    }
    return ret;
}
