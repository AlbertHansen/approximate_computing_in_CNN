#include <iostream>
#include <fstream>
#include <array>
#include <sstream>
#include <stdint.h>
#include <stdlib.h>
#include <cmath>
#include "Evaluator.h"

uint16_t add8u_6P8(uint8_t a, uint8_t b)
{
  uint16_t c = 0;
  uint8_t n0 = (a >> 0) & 0x1;
  uint8_t n2 = (a >> 1) & 0x1;
  uint8_t n4 = (a >> 2) & 0x1;
  uint8_t n6 = (a >> 3) & 0x1;
  uint8_t n8 = (a >> 4) & 0x1;
  uint8_t n10 = (a >> 5) & 0x1;
  uint8_t n12 = (a >> 6) & 0x1;
  uint8_t n14 = (a >> 7) & 0x1;
  uint8_t n16 = (b >> 0) & 0x1;
  uint8_t n18 = (b >> 1) & 0x1;
  uint8_t n20 = (b >> 2) & 0x1;
  uint8_t n22 = (b >> 3) & 0x1;
  uint8_t n24 = (b >> 4) & 0x1;
  uint8_t n26 = (b >> 5) & 0x1;
  uint8_t n28 = (b >> 6) & 0x1;
  uint8_t n30 = (b >> 7) & 0x1;
  uint8_t n33;
  uint8_t n35;
  uint8_t n37;
  uint8_t n38;
  uint8_t n43;
  uint8_t n46;
  uint8_t n50;
  uint8_t n56;
  uint8_t n58;
  uint8_t n59;
  uint8_t n68;
  uint8_t n69;
  uint8_t n77;
  uint8_t n78;
  uint8_t n79;
  uint8_t n84;
  uint8_t n86;
  uint8_t n87;
  uint8_t n92;
  uint8_t n96;
  uint8_t n97;
  uint8_t n110;
  uint8_t n122;
  uint8_t n127;
  uint8_t n134;
  uint8_t n152;
  uint8_t n162;
  uint8_t n171;
  uint8_t n180;
  uint8_t n198;
  uint8_t n199;
  uint8_t n226;
  uint8_t n240;
  uint8_t n244;
  uint8_t n254;
  uint8_t n272;
  uint8_t n391;
  uint8_t n394;
  uint8_t n404;
  uint8_t n412;
  uint8_t n413;
  uint8_t n423;

  n33 = n12 | n28;
  n35 = n0 | n16;
  n37 = ~(n4 & n20);
  n38 = ~n37;
  n43 = n38;
  n46 = ~(n18 | n2);
  n50 = n4 ^ n20;
  n56 = n43;
  n58 = (n6 ^ n22) ^ n56;
  n59 = (n6 & n22) | (n22 & n56) | (n6 & n56);
  n68 = n8 ^ n24;
  n69 = n8 & n24;
  n77 = n68;
  n78 = n10 ^ n26;
  n79 = n10 & n26;
  n84 = n59;
  n86 = n12 ^ n28;
  n87 = n12 & n28;
  n92 = n78;
  n96 = n14 ^ n30;
  n97 = n14 & n30;
  n110 = ~n46;
  n122 = n77;
  n127 = n69;
  n134 = n77 & n84;
  n152 = n127 | n134;
  n162 = n33 & n79;
  n171 = n86 & n78;
  n180 = n87 | n162;
  n198 = n152;
  n199 = n152;
  n226 = n92 & n199;
  n240 = ~n84;
  n244 = n79 | n226;
  n254 = n171 & n198;
  n272 = n180 | n254;
  n391 = ~(n240 ^ n122);
  n394 = n78 ^ n198;
  n404 = n86 ^ n244;
  n412 = n96 ^ n272;
  n413 = n96 & n272;
  n423 = n97 | n413;

  c |= (n35 & 0x1) << 0;
  c |= (n110 & 0x1) << 1;
  c |= (n50 & 0x1) << 2;
  c |= (n58 & 0x1) << 3;
  c |= (n391 & 0x1) << 4;
  c |= (n394 & 0x1) << 5;
  c |= (n404 & 0x1) << 6;
  c |= (n412 & 0x1) << 7;
  c |= (n423 & 0x1) << 8;

  return c;
}

constexpr size_t adderInputSize = 8;
constexpr size_t adderInputRange = pow(2,adderInputSize);
constexpr size_t adderOutputRange = pow(adderInputRange,2);

std::string FileName = "add8u_6P8.csv";

std::array<uint16_t, adderOutputRange> expected;
std::array<uint16_t, adderOutputRange> actual;
std::array<uint8_t, adderOutputRange> aInput;
std::array<uint8_t, adderOutputRange> bInput;

/**************************************************************************/

template<typename T, size_t N>
void writeArrayToCSV(const std::array<T, N>& arr, const std::string& filename) 
{
    std::ofstream outFile(filename);
    if (!outFile.is_open()) 
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    for (size_t i = 0; i < N; ++i) 
    {
        outFile << arr[i];  // Write the current element
        if (i != N - 1) 
        {
            outFile << ",";  // Add a comma if it's not the last element in the row
        }
    }

    outFile.close();
    std::cout << "Array written to " << filename << std::endl;
}

/**************************************************************************/

int main() {
    
    for (uint8_t a = 3; a < adderInputRange; a++)
    {
        
        for (uint8_t b = 3; b < adderInputRange; b++)
        {
            expected.at(adderInputRange*a+b) = add8u_6P8(a,b);
            actual.at(adderInputRange*a+b) = a+b;
            aInput.at(adderInputRange*a+b) = a;
            bInput.at(adderInputRange*a+b) = b;
            std::cout << "HEJ";
        }
    }
    writeArrayToCSV(expected, FileName);
    //Evaluator addu8_6P8_eval();
    return 0;
};