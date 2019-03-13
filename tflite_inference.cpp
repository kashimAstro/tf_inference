#include "tflite_inference.h"

template<typename T> void tflite_inference::fill(T *in, cv::Mat& src) {
   int n = 0, nc = src.channels(), ne = src.elemSize();
   for (int y = 0; y < src.rows; ++y)
      for (int x = 0; x < src.cols; ++x)
         for (int c = 0; c < nc; ++c)
            in[n++] = src.data[y * src.step + x * ne + c];
   }

int tflite_inference::setup(const string &modelfile, const string &labelfile) {
   if (modelfile == "" || labelfile == "") {
      cerr << "error: model | label empty" << endl;
      return -1;
      }
   model = tflite::FlatBufferModel::BuildFromFile(modelfile.c_str());
   if (!model) {
      cerr << "error load model." << endl;
      return -1;
      }
   
   tflite::InterpreterBuilder(*model, resolver)(&interpreter);
   status = interpreter->AllocateTensors();
   if (status != kTfLiteOk) {
      cerr << "error: allocate memory tesnors" << endl;
      return -1;
      }

   labels = tf_load_labels(labelfile.c_str()); 
   if(labels.empty()){
      cerr << "error: load labels vector" << endl;
      return -1;
      }
   return 0;
   }

vector< dati_inference > tflite_inference::inference(Mat frame, float threshold) {
   vector< dati_inference > result_inf;
   if(frame.empty()) {
      return result_inf;
   }
   int input = interpreter->inputs()[0];
   
   TfLiteIntArray* dims = interpreter->tensor(input)->dims;
   int wanted_height = dims->data[1];
   int wanted_width = dims->data[2];
   int wanted_channels = dims->data[3];
   int wanted_type = interpreter->tensor(input)->type;
   
   uint8_t *in8 = nullptr;
   float *in16 = nullptr;
   
   if (wanted_type == kTfLiteFloat32) {
      in16 = interpreter->typed_tensor<float>(input);
      }
   else if (wanted_type == kTfLiteUInt8) {
      in8 = interpreter->typed_tensor<uint8_t>(input);
      }
   
   interpreter->SetNumThreads(4);
   interpreter->UseNNAPI(1);
   
   Mat resized(wanted_height, wanted_width, frame.type());
   resize(frame, resized, resized.size(), INTER_CUBIC);
   int n = 0;
   
   if (wanted_type == kTfLiteFloat32) {
      fill(in16, resized);
      }
   else if (wanted_type == kTfLiteUInt8) {
      fill(in8, resized);
      }
   
   status = interpreter->Invoke();
   if (status != kTfLiteOk) {
      cerr << "error: invoke interpreter" << endl;
      return result_inf;
      }
   
   int output = interpreter->outputs()[0];
   cerr << "output: " << output << endl;
   
   TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;
   auto output_size = output_dims->data[output_dims->size - 1];
   int output_type = interpreter->tensor(output)->type;
   std::vector<std::pair<float, int>> results;
   
   if (wanted_type == kTfLiteFloat32) {
      float *scores = interpreter->typed_output_tensor<float>(0);
      for (int i = 0; i < output_size; ++i) {
         float value = (scores[i] - 127) / 127.0;
         if (value > threshold)
         	results.push_back(std::pair<float, int>(value, i));
         }
      }
   else if (wanted_type == kTfLiteUInt8) {
      uint8_t *scores = interpreter->typed_output_tensor<uint8_t>(0);
      for (int i = 0; i < output_size; ++i) {
         float value = (float)scores[i] / 255.0;
         if (value > threshold)
         	results.push_back(std::pair<float, int>(value, i));
         }
      }
   std::sort(results.begin(), results.end(), [](std::pair<float, int>& x, std::pair<float, int>& y) -> int { return x.first > y.first; } );
   
   for (const auto& result : results) {
      dati_inference res;
      res.id    = result.second;
      res.prob  = result.first;
      res.label = labels[result.second];
      result_inf.push_back(res);
      }
   return result_inf;
   }

vector<string> tflite_inference::tf_load_labels(const char * file_name) {
   vector<string> d;
   std::ifstream file(file_name);
   if (!file) {
      return d;
      }
   std::string line;
   while (std::getline(file, line)) {
      d.push_back(line);
      }

   const int padding = 16;
   while (d.size() % padding) {
      d.emplace_back();
      }
   return d;
   }

