#include <tensorflow/core/lib/histogram/histogram.h>
#include <tensorflow/core/util/events_writer.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <string>
#include <iostream>
#include <float.h>

class tf_tensorboard {
  public:
    void write_histogram(tensorflow::EventsWriter* writer, double wall_time, tensorflow::int64 step, const std::string& tag, tensorflow::HistogramProto *hist);
    void write_audio(tensorflow::EventsWriter* writer, double wall_time, tensorflow::int64 step, const std::string& tag, tensorflow::Summary_Audio* audio);
    void write_tensor(tensorflow::EventsWriter* writer, double wall_time, tensorflow::int64 step, const std::string& tag, tensorflow::TensorProto* tensor);
    void write_scalar(tensorflow::EventsWriter* writer, double wall_time, tensorflow::int64 step, const std::string& tag, float simple_value);
};
