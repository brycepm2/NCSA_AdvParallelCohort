#include <iostream>
#include <vector>
#include <complex>
#include <sys/time.h>
#include "omp.h"

#define ORD 1 << 27

using std::cout;
using std::endl;

typedef std::complex<float> cFloat;

int mandelbrot(cFloat& c);

int main() {
  // instantiate vector of complex numbers
  std::vector<cFloat> v;
  float a, b;
  srand(time(NULL));
  for (int iPos = 0; iPos < ORD; iPos++) {
    // make 2 random doubles
    a = rand() % RAND_MAX;
    a = float(a) / (RAND_MAX);
    b = rand() % RAND_MAX;
    b = float(b) / (RAND_MAX);
    // make our complex number
    cFloat z(a, b);
    // add it to the vector
    v.push_back(z);
  }

  // stopwatch variables
  struct timeval startTime, stopTime, elapsedTime;

  gettimeofday(&startTime, NULL);
  // note: must initialize iterator in loop
  // or code segFaults
  #pragma omp parallel
  {
    #pragma omp for
    for (std::vector<cFloat>::iterator vPos = v.begin(); vPos != v.end(); ++vPos) {
      // calculate setVal
      const int iTs = mandelbrot(*vPos);
    }
  }

  gettimeofday(&stopTime, NULL);

  timersub(&stopTime, &startTime, &elapsedTime);
  cout << "Number of args: " << v.size() << " Elapsed time (s): "
       << elapsedTime.tv_sec + elapsedTime.tv_usec / 1000000.0 << endl;

  return 0;
}

int mandelbrot(cFloat &c) {
  cFloat z(0.0, 0.0);
  const unsigned int maxIt = 200;
  for (unsigned int iMan = 0; iMan < maxIt; iMan++) {
    if (std::abs(z) > 2.0) {
      // outside of magnitude bounds
      return iMan;
    }
    // square z
    z = z * z + c;
  }
  return maxIt;
}
