#include <iostream>
#include <vector>

std::vector<intmax_t> expected = {2};
std::vector<intmax_t> actual = {4};

int main()
{
    intmax_t xorResult = expected[0] ^ actual[0];
    std::cout << xorResult;
};