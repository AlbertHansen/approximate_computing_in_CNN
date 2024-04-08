#include <iostream>
#include <fstream>
#include <array>
#include <sstream>
#include <stdint.h>
#include <stdlib.h>
#include <cmath>
#include "Evaluator.h"

uint64_t add8u_00M(uint64_t a, uint64_t b) {
  int wa[8];
  int wb[8];
  uint64_t y = 0;
  wa[0] = (a >> 0) & 0x01;
  wb[0] = (b >> 0) & 0x01;
  wa[1] = (a >> 1) & 0x01;
  wb[1] = (b >> 1) & 0x01;
  wa[2] = (a >> 2) & 0x01;
  wb[2] = (b >> 2) & 0x01;
  wa[3] = (a >> 3) & 0x01;
  wb[3] = (b >> 3) & 0x01;
  wa[4] = (a >> 4) & 0x01;
  wb[4] = (b >> 4) & 0x01;
  wa[5] = (a >> 5) & 0x01;
  wb[5] = (b >> 5) & 0x01;
  wa[6] = (a >> 6) & 0x01;
  wb[6] = (b >> 6) & 0x01;
  wa[7] = (a >> 7) & 0x01;
  wb[7] = (b >> 7) & 0x01;
  int sig_16 = wa[6] & wb[6];
  int sig_26 = 0;
  int sig_32 = ~(wa[6] | wb[6]);
  int sig_45 = wb[7];
  int sig_47 = wb[6] | wa[6];
  int sig_48 = wa[7] ^ wb[7];
  int sig_49 = wa[7] & wb[7];
  int sig_50 = sig_48 & sig_47;
  int sig_51 = sig_48 ^ sig_47;
  int sig_52 = sig_49 | sig_50;
  y |=  (sig_52 & 0x01) << 0; // default output
  y |=  (wa[3] & 0x01) << 1; // default output
  y |=  (wa[7] & 0x01) << 2; // default output
  y |=  (sig_26 & 0x01) << 3; // default output
  y |=  (sig_45 & 0x01) << 4; // default output
  y |=  (sig_16 & 0x01) << 5; // default output
  y |=  (sig_32 & 0x01) << 6; // default output
  y |=  (sig_51 & 0x01) << 7; // default output
  y |=  (sig_52 & 0x01) << 8; // default output
   return y;
}

constexpr size_t adderInputSize = 8;
constexpr size_t adderInputRange = pow(2,adderInputSize);
constexpr size_t adderOutputRange = pow(adderInputRange,2);

std::string expectedFile = "expectedFile.csv";
std::string actualFile = "actualFile.csv";
std::string aFile = "aFile.csv";
std::string bFile = "bFile.csv";

std::array<uint16_t, adderOutputRange+1> expected;
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
    
    for (uint16_t a = 0; a < adderInputRange; a++)
    {
        
        for (uint16_t b = 0; b < adderInputRange; b++)
        {
            expected.at(adderInputRange*a+b) = add8u_00M(a,b);
            actual.at(adderInputRange*a+b) = a+b;
            aInput.at(adderInputRange*a+b) = a;
            bInput.at(adderInputRange*a+b) = b;
            //std::cout << "HEJ";
        }
    }
    writeArrayToCSV(expected, expectedFile);
    writeArrayToCSV(actual, actualFile);
    writeArrayToCSV(aInput, aFile);
    writeArrayToCSV(bInput, bFile);
    //Evaluator addu8_6P8_eval();
    return 0;
};