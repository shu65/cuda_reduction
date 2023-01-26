#include <stdint.h>
#include <cstddef>

#include "reduction.hpp"

using namespace std;

int reduce(const int *in, size_t n)
{
    int ret = 0;
    for (size_t i = 0; i < n; ++i)
    {
        ret += in[i];
    }
    return ret;
}
