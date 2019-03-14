#include "tf_tensorboard.h" 

void tf_tensorboard::write_histogram(tensorflow::EventsWriter* writer, double wall_time, tensorflow::int64 step, const std::string& tag, tensorflow::HistogramProto *hist) {
  tensorflow::Event event;
  event.set_wall_time(wall_time);
  event.set_step(step);
  tensorflow::Summary::Value* summ_val = event.mutable_summary()->add_value();
  summ_val->set_tag(tag);
  summ_val->set_allocated_histo(hist);
  writer->WriteEvent(event);
}

void tf_tensorboard::write_audio(tensorflow::EventsWriter* writer, double wall_time, tensorflow::int64 step, const std::string& tag, tensorflow::Summary_Audio* audio) {
  tensorflow::Event event;
  event.set_wall_time(wall_time);
  event.set_step(step);
  tensorflow::Summary::Value* summ_val = event.mutable_summary()->add_value();
  summ_val->set_tag(tag);
  summ_val->set_allocated_audio(audio);
  writer->WriteEvent(event);
}

void tf_tensorboard::write_tensor(tensorflow::EventsWriter* writer, double wall_time, tensorflow::int64 step, const std::string& tag, tensorflow::TensorProto* tensor) {
  tensorflow::Event event;
  event.set_wall_time(wall_time);
  event.set_step(step);
  tensorflow::Summary::Value* summ_val = event.mutable_summary()->add_value();
  summ_val->set_tag(tag);
  summ_val->set_allocated_tensor(tensor);
  writer->WriteEvent(event);

}

void tf_tensorboard::write_scalar(tensorflow::EventsWriter* writer, double wall_time, tensorflow::int64 step, const std::string& tag, float simple_value) {
  tensorflow::Event event;
  event.set_wall_time(wall_time);
  event.set_step(step);
  tensorflow::Summary::Value* summ_val = event.mutable_summary()->add_value();
  summ_val->set_tag(tag);
  summ_val->set_simple_value(simple_value);
  writer->WriteEvent(event);
}
