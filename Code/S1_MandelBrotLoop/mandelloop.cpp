 #include <iostream>
 #include <vector>
 #include <complex>
 #include <sys/time.h>

 #define ORD 1 << 3

 typedef std::complex<float> cFloat;

int mandelbrot(cFloat& c);

 int main() {
   // instantiate vector of
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
     cFloat z (a, b);
     // add it to the vector
     v.push_back(z);
   }

   std::vector<cFloat>::iterator vPos = v.begin();
   std::vector<cFloat>::iterator vEnd = v.end();
   for ( ; vPos != vEnd; vPos++) {
     std::cout << "vPos = " << (*vPos) << std::endl;
     // calculate setVal
     const int iTs = mandelbrot( *vPos);
     std::cout << "iTs = " << iTs << std::endl;
   }
   return 0;
 }

 int mandelbrot(cFloat& c) {
   cFloat z(0.0, 0.0);
   const unsigned int maxIt = 200;
   for (unsigned int iMan = 0; iMan < maxIt; iMan++) {
     if ( std::abs(z) > 2.0 ) {
       // outside of magnitude bounds
       std::cout << "z = " << z << std::endl;
       return iMan;
     }
     // square z
     z = z*z + c;
   }
   std::cout << "z = " << z << std::endl;
   return maxIt;
 }
