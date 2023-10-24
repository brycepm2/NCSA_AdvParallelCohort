// Bryce Mazurowski (2023)
// brycepm2@gmail.com

#include <iostream>
#include <vector>
#include <map>
#include <math.h>

#include <omp.h>

using std::cout;
using std::endl;
using std::map;
using std::vector;

struct stateVar {
  // default constructor
  stateVar()=default;
  // default destructor
  ~stateVar()=default;
  // default copy constructor
  stateVar(stateVar& svIn)=default;

  // vector to store info for each intPt
  std::vector<double> vals;
};


// how to print data structure
std::ostream& operator<<(std::ostream& os, const stateVar& svObj) {
  cout << "stateVar: " << endl;
  std::vector<double> valsOut = svObj.vals;
  std::vector<double>::iterator iter_val = valsOut.begin();
  for (; iter_val != valsOut.end(); ++iter_val) {
    cout << (*iter_val) << ' ';
  }
  return os;
}

// map for each element
typedef map<int, stateVar*> stateVMap;

/** @brief loop over integration points insert number
 * into data structure for each int point
 */
void integrateLoop(stateVar* p_elemStateVar);

int main() {
  // set loop limits
  const int loopEnd = 1 << 4;

  // instantiate stateVarMap
  stateVMap svMap;
  double sTime = omp_get_wtime();
  #pragma omp parallel
  {
    #pragma omp for
    for (int iElem = 0; iElem < loopEnd; ++iElem) {
      // create stateVar data structure for each elem
      stateVar* p_elemStateVar = new stateVar();
      // run integration loop
      integrateLoop(p_elemStateVar);
      // insert data structure into map
      svMap.insert({iElem, p_elemStateVar});
    }
  }
  double eTime = omp_get_wtime();

  cout << "Total time (s): " << (eTime - sTime) << endl;

  // loop over map and print each point
  stateVMap::iterator iter_svMap = svMap.begin();
  for (; iter_svMap != svMap.end(); ++iter_svMap) {
    cout << "Key: " << iter_svMap->first << endl;
    cout << "struct: " << (*iter_svMap->second) << endl;
  }
  return 0;
}


void integrateLoop(stateVar* p_elemStateVar) {
  // number of integration points
  const int nPts = 18;
  for (int iPt = 0; iPt < nPts; ++iPt) {
    double x = double(iPt);
    // make value some function
    const double value = std::pow(x,3.0) + 5*x*x + 13;
    // add value to data structure
    p_elemStateVar->vals.push_back(value);
  }
} 
