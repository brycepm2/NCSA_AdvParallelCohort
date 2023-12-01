// Iterative linear solver I intend to MPIify
// By Bryce Mazurowski
//

#include <iostream>
#include <vector>
#include <sys/time.h>
#include <math.h>

using std::cout;
using std::endl;
using std::sqrt;
using std::vector;

void matVecProd(const vector<vector<double>>& A,
                const vector<double>& x,
                const double alpha,
                const char trans,
                vector<double>& out);

void vecMinVec(const vector<double>& a,
               const vector<double>& b,
               vector<double>& c);

void dotProd(const vector<double>& a,
             const vector<double>& b,
             double& c);

void gradDescUpdate(const vector<double>& grad,
                    const double learnRate,
                    vector<double>& xNew);

void printVec(const vector<double>& v);

int main() {
  // system size
  int n = 4;
  // A matrix
  vector<vector<double>> A(n, vector<double>(n,0.0));
  // x vector
  vector<double> x(n, 0.0);
  // rhs
  vector<double> b(n, 0.0);
  // residual vector
  vector<double> r(n, 0.0);

  struct timeval sTime, eTime, elapsedTime;
  gettimeofday(&sTime, NULL);
  // random doubles to populate system
  double y, z;
  srand(time(NULL));
  for (int iRow = 0; iRow < n; iRow++) {
    for (int jCol = 0; jCol < n; jCol++) {
      // make random doubles
      y = rand() % RAND_MAX;
      y = float(y) / (RAND_MAX);
      z = rand() % RAND_MAX;
      z = float(z) / (RAND_MAX);
      A[iRow][jCol] = y;
      b[iRow] = z;
    }
  }
  // cout << "b = " << endl;
  // printVec(b); 
  const int nRuns = 5000;
  vector<double> xOut(n, 0.0);
  for (int iPass = 0; iPass <= nRuns; ++iPass) {
    // FORWARD PASS
    // MatrixVector Prod
    matVecProd(A, x, 1.0 /*alpha*/, 'n', xOut);

    // Make residual
    vecMinVec(b, xOut, r);
    // cout << "r = " << endl;
    // printVec(r);

    // ||r||_{2} = \sqrt(r.r)
    double loss = 0.0;
    dotProd(r, r, loss);
    loss = sqrt(loss);
    if (!(iPass % 500)) {
      cout << "Run: " << iPass
           << " LOSS = " << loss << endl;
    }

    // BACKWARD PASS
    // dL/dr_i * dr_i/dx_j = -A_{ji} r_i/loss = -1/loss*a_{ji} r_i
    // dL/dr_i = r_i/loss
    // dr_i/dx_j = -A_{ji}
    vector<double> grad(n, 0.0);
    matVecProd(A, r, -1.0 / loss /*alpha*/, 't', grad);
    // cout << "grad = " << endl;
    // printVec(grad);

    // gradient descent
    double learnRate = 0.001;
    gradDescUpdate(grad, learnRate, x);
    // cout << "x = " << endl;
    // printVec(x);
  }
  // test results
  cout << "x = " << endl;
  printVec(x);
  matVecProd(A, x, 1.0, 'n', xOut);
  vecMinVec(b, xOut, r);
  cout << "Final r = " << endl;
  printVec(r);
  gettimeofday(&eTime, NULL);
  timersub(&eTime, &sTime, &elapsedTime);
  cout << "Elapsed time (s): "  
       << elapsedTime.tv_sec + elapsedTime.tv_usec / 1000000.0 << endl;

  return 0;
}

void matVecProd(const vector<vector<double>>& A,
                const vector<double>& x,
                const double alpha,
                const char trans,
                vector<double>& out) {
  // alpha*A.x
  // trans determines whether A_{ij} or A_{ji}
  // output into nonConst vector
  // zero out
  std::fill(out.begin(), out.end(), 0.0);
  int size = x.size();
  double aComp = 0.0;
  for (int iRow = 0; iRow < size; ++iRow) {
    for (int jCol = 0; jCol < size; ++jCol) {
      if (trans == 'n') {
        aComp = A[iRow][jCol];
      } else {
        aComp = A[jCol][iRow];
      }
      out[iRow] += aComp*x[jCol];
    }
    out[iRow] *= alpha;
  }
}

void vecMinVec(const vector<double>& a,
               const vector<double>& b,
               vector<double>& c) {
  // a_i - b_i = c_i
  for (int iRow = 0; iRow < a.size(); ++iRow) {
    c[iRow] = a[iRow] - b[iRow];
  }
}

void dotProd(const vector<double>& a,
             const vector<double>& b,
             double& c) {
  // a_i . b_i = c
  for (int iRow = 0; iRow < a.size(); ++iRow) {
    c += a[iRow]*b[iRow];
  }
}

void gradDescUpdate(const vector<double>& grad,
                    const double learnRate,
                    vector<double>& xNew) {
  // xNew_i = xNew_i - grad*learnRate
  for (int iRow = 0; iRow < grad.size(); ++iRow) {
    xNew[iRow] -= grad[iRow]*learnRate;
  }
}

void printVec(const vector<double>& v) {
  for (int iRow = 0; iRow < v.size(); ++iRow) {
    std::cout << v[iRow] << " ";
  }
  std::cout << std::endl;
}
 
