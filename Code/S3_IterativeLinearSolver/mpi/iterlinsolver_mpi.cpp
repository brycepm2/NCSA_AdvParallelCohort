// Iterative linear solver I intend to MPIify
// By Bryce Mazurowski
//

#include <iostream>
#include <vector>
#include <sys/time.h>
#include <cmath>

#include "mpi.h"

#define ORD 512

using std::cout;
using std::endl;
using std::sqrt;
using std::vector;

void matVecProd(const vector<double> A,
                const int chunkSize,
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
  int locRank, globRank, mpiErr;
  // local and global loss
  double loss, globLoss;
  // gradient descent rate
  double learnRate = 0.0001;

  // Each threads components
  // A matrix - NOTE: MPI seems to hate vector<vector>
  vector<double> A;
  // x vector - Need full vector in initial split
  vector<double> x;
  vector<double> xTest;
  // work area
  vector<double> xOut;
  // rhs
  vector<double> b;
  // residual vector
  vector<double> r;
  // vector for gradient
  vector<double> grad;
  int rowStart, rowEnd;
  int chunkSize, leftOver;

  // full thread components
  vector<double> fullGrad(ORD);
  vector<double> fullRes(ORD);
  // For testing
  vector<double> globA(ORD*ORD);
  vector<double> globX(ORD);
  vector<double> globB(ORD);

  vector<double> lossCheckVec(ORD);

  // random doubles to populate system
  double y, z;


  double sTime, eTime;
  mpiErr = MPI_Init(NULL, NULL);
  
  // This rank
  MPI_Comm_rank(MPI_COMM_WORLD, &locRank);
  // global rank
  MPI_Comm_size(MPI_COMM_WORLD, &globRank);

  if (locRank == 0) {
    sTime = MPI_Wtime();
  } 
 
  // To start:
  // cut up A, x, b by rowWise
  chunkSize = ORD/globRank;
  leftOver = ORD % globRank;
  if (locRank == (globRank - 1)) {
    chunkSize += leftOver;
  }
  // resize this threads vectors
  A.resize(chunkSize*ORD);
  x.resize(ORD);
  xTest.resize(chunkSize);
  xOut.resize(chunkSize);
  b.resize(chunkSize);
  r.resize(chunkSize);
  grad.resize(chunkSize);

  srand(10);
  if (locRank == 0) {
    for (int iRow = 0; iRow < ORD; ++iRow) {
      for (int jCol = 0; jCol < ORD; ++jCol) {
      // make random doubles
      y = rand() % RAND_MAX;
      y = float(y) / (RAND_MAX);
      globA[iRow*ORD + jCol] = y;
      }
      z = rand() % RAND_MAX;
      z = float(z) / (RAND_MAX);
      globX[iRow] = z;
      x[iRow] = 0.0;
    }
  }

  // each thread fills in their rows
  for (int iRow = 0; iRow < chunkSize; iRow++) {
    // initialize some things
    xOut[iRow] = 0.0;
    r[iRow] = 0.0;
    grad[iRow] = 0.0;
  }
  // Make manufactured solution
  if (locRank == 0) {
    matVecProd(globA, ORD, globX, 1.0 /*alpha*/, 'n', globB);
    // cout << "globA:" << endl;
    // printVec(globA);
    // cout << "globB:" << endl;
    // printVec(globB);
  }
  // Push rhs out to everyone
  mpiErr = MPI_Scatter(globB.data(), chunkSize, MPI_DOUBLE, b.data(),
                       chunkSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  // kick out little slices of globA to each proc
  mpiErr = MPI_Scatter(globA.data(), ORD*chunkSize, MPI_DOUBLE, A.data(),
                      ORD*chunkSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  const int nRuns = 5000;
  for (int iPass = 0; iPass <= nRuns; ++iPass) {
    // FORWARD PASS
    // MatrixVector Prod
    matVecProd(A, chunkSize, x, 1.0 /*alpha*/, 'n', xOut);

    // Make residual
    vecMinVec(b, xOut, r);
    
    // ||r||_{2} = \sqrt(r.r)
    dotProd(r, r, loss);
    mpiErr = MPI_Allreduce(&loss, &globLoss, 1, MPI_DOUBLE,
                           MPI_SUM, MPI_COMM_WORLD);
    globLoss = sqrt(globLoss);
    if (!(iPass % 1000) && locRank == 0) {
      cout << "Run: " << iPass
           << " LOSS = " << globLoss << endl;
    }

    // BACKWARD PASS
    // dL/dr_i * dr_i/dx_j = -A_{ji} r_i/loss = -1/loss*a_{ji} r_i
    // dL/dr_i = r_i/loss
    // dr_i/dx_j = -A_{ji}
    // NOTE: this needs to be global :(
    MPI_Gather(r.data(), chunkSize, MPI_DOUBLE, fullRes.data(),
               chunkSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (locRank == 0) {
      matVecProd(globA, ORD, fullRes, -1.0/globLoss /*alpha*/, 't', fullGrad);
      // update x vector
      gradDescUpdate(fullGrad, learnRate, x);
    }
 
    // Update x for all threads
    MPI_Bcast(x.data(), x.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  matVecProd(A, chunkSize, x, 1.0, 'n', xOut);
  vecMinVec(b, xOut, r);
  mpiErr = MPI_Gather(r.data(), chunkSize, MPI_DOUBLE, fullRes.data(),
                      chunkSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if (locRank == 0) {
    // test results
    // cout << "manufact sol = " << endl;
    // printVec(globX);
    // cout << "final x = " << endl;
    // printVec(x);
    double testLoss = 0.0;
    vecMinVec(globX, x, lossCheckVec);
    dotProd(lossCheckVec, lossCheckVec, testLoss);
    testLoss = sqrt(testLoss);
    cout << "L2 Error = " << testLoss << endl;
    double finLoss = 0.0;
    dotProd(fullRes, fullRes, finLoss);
    finLoss = sqrt(finLoss);
    cout << "Final loss = " << finLoss << endl;
    eTime = MPI_Wtime();
    cout << "Elapsed time (s): "  
         << eTime - sTime << endl;
  }
  mpiErr = MPI_Finalize();
  return 0;
}

void matVecProd(const vector<double> A,
                const int chunkSize,
                const vector<double>& x,
                const double alpha,
                const char trans,
                vector<double>& out) {
  // alpha*A.x
  // trans determines whether A_{ij} or A_{ji}
  // output into nonConst vector
  // zero out
  std::fill(out.begin(), out.end(), 0.0);
  int nRow = chunkSize;
  int nCol = x.size();
  double aComp = 0.0;
  for (int iRow = 0; iRow < nRow; ++iRow) {
    for (int jCol = 0; jCol < nCol; ++jCol) {
      if (trans == 'n') {
        aComp = A[iRow*nCol + jCol];
      } else {
        aComp = A[jCol*nCol + iRow];
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
  c = 0.0;
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
 
