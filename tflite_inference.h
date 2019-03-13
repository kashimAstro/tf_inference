#include <vector>
#include <chrono>
#include <iostream>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

#include <tensorflow/contrib/lite/model.h>
#include <tensorflow/contrib/lite/interpreter.h>
#include <tensorflow/contrib/lite/kernels/register.h>

using namespace std;
using namespace cv;
using namespace tflite;

struct dati_inference {
        int id;
        string label;
        float prob;
        Rect rect;
};

class tflite_inference {
    public:
    template<typename T> void fill(T *in, cv::Mat& src);
    int setup(const string &modelfile, const string &labelfile);
    std::vector<dati_inference> inference(Mat frame, float threshold=0.0001);
    private:
    TfLiteStatus status;
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    vector<string> labels;
    vector<string> tf_load_labels(const char * file_name);
};
