#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <cstring>
#include "common.h"
using namespace std;

Common::PerformanceTimer& timer();

struct Params {
  vector<int> layer_sizes;
  int layer_count, input_size;
  int output_size;
};

class Net {
public:
  Net(int n, vector<int> layers);
  double* forward(double *data, int n);
  void load_weights(string path);
private:
  double *data;
  double *y;
  double *exp_data;

  vector<double *> w;
  vector<double *> b;
  vector<double *> z; // z = w * x + b
  vector<double *> a; //
  // parameters
  Params params;
  void fill_rand(double *A, int size, double std);
  void matmul(const double *A, const double *B, double *C, const int M, const int K, const int N);
  void bias_addition(int n, double *idata, double *bias, double *odata);
  void relu_activation(int n, double *idata, double *odata);
  void broadcast_sub(int n, double *idata, double *odata, double sub);
  void elementwise_exp(int n, double *idata, double *odata);
  void reduce_max(int n, double *idata, double *odata);
  void reduce_sum(int n, double *idata, double *odata);
  void broadcast_div(int n, double *idata, double *odata, double exp_sum);
};