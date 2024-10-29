#include <string>
#include <iostream>
#include <filesystem>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "mlp.h"

#define classes 52
#define in_dim 15
#define epochs 400
#define inputs in_dim*in_dim

using namespace std;

float total_time_forward = 0;

int image_read(string path, vector<double *> &data) {
  cv::Mat image = cv::imread(path.c_str(), cv::IMREAD_GRAYSCALE);
  if (!image.data)
  {
    cout << "Could not open or find the image" << std::endl;
    return 0;
  }
  data.push_back(new double[inputs]);
  cv::resize(image, image, cv::Size(in_dim, in_dim));
  for (int i = 0; i < inputs; i++)
    data[data.size() - 1][i] = (double)image.data[i];
  return 1;
}

void lable_read(string name, vector<double *> &data) {
  int value = stoi(name.substr(0, 2));
  data.push_back(new double[classes]);
  memset(data[data.size() - 1], 0, classes * sizeof(double));
  data[data.size() - 1][value - 1] = 1;
}

void read_directory(const std::string& name, vector<string>& v) {
  for (const auto& entry : std::filesystem::directory_iterator(name)) {
    if (entry.is_regular_file() && entry.path().extension() == ".bmp") {
      v.push_back(entry.path().filename().string());
    }
  }
}

int main(int argc, char* argv[]) {
  string path = "./data-set/";
  vector<string> files;
  vector<double *> input_data;
  vector<double *> output_data;

  read_directory(path, files);
  for (auto x : files) {
    if (image_read(path + x, input_data)) {
      lable_read(x, output_data);
    }
  }

  // forward pass
  Net nn(inputs, {98, 65, 50, 30, 25, 40, classes});
  nn.load_weights("./weights.csv");
  int i;
  int val = 0;
  for (i = 0; i < epochs; i++) {
    total_time_forward += timer().getCpuElapsedTimeForPreviousOperation();
    for (int i = 0; i < classes; i++) {
      double* x = nn.forward(input_data[i], inputs);
      int pos1 = distance(x, max_element(x, x + classes));
      int pos2 = distance(output_data[i], max_element(output_data[i], output_data[i] + classes));
      val += pos1 == pos2;
      delete[] x;
    }
  }
  cout << "Inference total forward = " << total_time_forward << " ms, avg forward = " << total_time_forward / epochs << " ms" << endl;
  cout << "Accuracy : " << (float)val / (epochs * classes) * 100.0 << "%" << endl;
}