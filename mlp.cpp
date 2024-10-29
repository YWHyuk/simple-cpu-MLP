#include "mlp.h"

Common::PerformanceTimer& timer()
{
  static Common::PerformanceTimer timer;
  return timer;
}

void Net::fill_rand(double *A, int size, double std) {
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.0, std);
  for (int i=0; i<size; i++) {
    A[i] = distribution(generator);
  }
}

Net::Net(int n, vector<int> layers) {
  params.layer_count = layers.size();
  params.input_size = n;
  params.output_size = layers[params.layer_count - 1];
  params.layer_sizes = layers;

  data = (double*)(malloc(n * sizeof(double)));
  y = (double*)(malloc(n * sizeof(double)));
  exp_data = (double*)(malloc(params.output_size * sizeof(double)));
 
  // add input layer to front
  layers.insert(layers.begin(), n);
  double *layer_w, *layer_b, *layer_z, *layer_a;
  for (int i = 0; i < params.layer_count; i++) {
    layer_w = (double*)(malloc(layers[i] * layers[i + 1] * sizeof(double)));
    layer_b = (double*)(malloc(layers[i+1] * sizeof(double)));

    // initilize w, b using gaussian distribution
    fill_rand(layer_w, layers[i] * layers[i + 1], 2.0 / (layers[i])); // uniform random initilization
    for (int j=0; j<layers[i + 1]; j++){
      layer_b[j] = 0.1;
    }
    w.push_back(layer_w);
    b.push_back(layer_b);

    // intermediate results arrays
    layer_z = (double*)(malloc(layers[i + 1] * sizeof(double)));
    layer_a = (double*)(malloc(layers[i + 1] * sizeof(double)));
    z.push_back(layer_z);
    a.push_back(layer_a);
  }
}

void Net::load_weights(string path) {
    std::ifstream infile(path);
    if (!infile.is_open()) {
        std::cerr << "Could not open file: " << path << std::endl;
        return;
    }

    std::string line;
    for (int i = 0; i < params.layer_count; i++) {
      int lmi;
      if (i != 0)
        lmi = params.layer_sizes[i - 1];
      else
        lmi = params.input_size;

      int n = params.layer_sizes[i] * lmi;
      double *temp_weights = new double[n];

      // Find the weight header for this layer
      std::getline(infile, line);  // W[i] header
      std::getline(infile, line);  // "-----" separator

      // Read weights
      for (int j = 0; j < n && std::getline(infile, line); j++) {
        temp_weights[j] = std::stod(line);
      }
      memcpy(w[i], temp_weights, n * sizeof(double));
      delete[] temp_weights;
    }

    for (int i = 0; i < params.layer_count; i++) {
      int n = params.layer_sizes[i];
      double *temp_biases = new double[n];

      // Find the bias header for this layer
      std::getline(infile, line);  // b[i] header
      std::getline(infile, line);  // "-----" separator

      // Read biases
      for (int j = 0; j < n && std::getline(infile, line); j++) {
        temp_biases[j] = std::stod(line);
      }
      memcpy(b[i], temp_biases, n * sizeof(double));
      delete[] temp_biases;
    }

    infile.close();
}

void Net::matmul(const double *A, const double *B, double *C, const int M, const int K, const int N) {
  for (int m=0; m<M; m++) {
    for (int n=0; n<N; n++) {
      int c_index = n * M + m;
      C[c_index] = 0.0;
      for (int k=0; k<K; k++) {
        int a_index = k * M + m;
        int b_index = n * K + k;
        C[c_index] += A[a_index] * B[b_index];
      }
    }
  }
}

void Net::bias_addition(int n, double *idata, double *bias, double *odata) {
  for (int i=0; i<n; i++) {
    odata[i] = idata[i] + bias[i];
  }
}

void Net::relu_activation(int n, double *idata, double *odata) {
  for (int i=0; i<n; i++) {
    odata[i] = idata[i] > 0.0 ? idata[i] : 0.0;
  }
}

void Net::broadcast_sub(int n, double *idata, double *odata, double sub) {
  for (int i=0; i<n; i++) {
    odata[i] = idata[i] - sub;
  }
}

void Net::elementwise_exp(int n, double *idata, double *odata) {
  for (int i=0; i<n; i++) {
    odata[i] = std::exp(idata[i]);
  }
}

void Net::reduce_max(int n, double *idata, double *odata) {
  for (int i=0; i<n; i++) {
    *odata = *odata > idata[i] ? *odata : idata[i];
  }
}

void Net::reduce_sum(int n, double *idata, double *odata) {
  for (int i=0; i<n; i++) {
    *odata += idata[i];
  }
}

void Net::broadcast_div(int n, double *idata, double *odata, double exp_sum) {
  for (int i=0; i<n; i++) {
    odata[i] = idata[i] / exp_sum;
  }
}

double* Net::forward(double *data, int n) {
  timer().startCpuTimer();
  double exp_sum;
  double out_max;
  double *res = new double[params.output_size]();

  for (int i = 0; i < params.layer_count; i++) {
    // matrix multiplication
    if (!i) { // first iteration, so a[i-1] hasn't been set yet
      matmul(w[i], data, z[i], params.layer_sizes[i], params.input_size, 1);
    }
    else {
      matmul(w[i], a[i - 1], z[i], params.layer_sizes[i], params.layer_sizes[i - 1], 1);
    }

    // bias addition
    bias_addition(params.layer_sizes[i], z[i], b[i], z[i]);
    if (i != params.layer_count - 1) {
      // relu activation
      relu_activation(params.layer_sizes[i], z[i], a[i]);
    }
    else {
      out_max = z[i][0];
      reduce_max(params.layer_sizes[i], z[i], &out_max);
      broadcast_sub(params.layer_sizes[i], z[i], z[i], out_max);
      elementwise_exp(params.layer_sizes[i], z[i], exp_data);
      exp_sum = 0;
      reduce_sum(params.layer_sizes[i], exp_data, &exp_sum);
      broadcast_div(params.layer_sizes[i], exp_data, a[i], exp_sum);
    }
  }
  memcpy(res, a[params.layer_count - 1], params.layer_sizes[params.layer_count - 1] * sizeof(double));
  timer().endCpuTimer();
  return res;
}
