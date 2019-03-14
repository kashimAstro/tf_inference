#include "tf_tensorboard.h"

int main(int argc, char const *argv[]) {
  std::string envent_file = "./events";
  tensorflow::EventsWriter writer(envent_file);
  tf_tensorboard board;

  for (int time_step = 0; time_step < 150; ++time_step) {

    //histogram semplicissimo 
    tensorflow::histogram::Histogram h;
    for (int i = 0; i < time_step; i++)
      h.Add(i);

    // conversione in formato proto
    tensorflow::HistogramProto *hist_proto = new tensorflow::HistogramProto();
    h.EncodeToProto(hist_proto, true);

    // salvo il grapo 
    board.write_histogram(&writer, time_step * 20, time_step, "some_hist", hist_proto);
    board.write_scalar(&writer, time_step * 20, time_step, "loss", time_step);

    /* bug
    tensorflow::TensorProto *tproto;
    tproto->set_dtype(tensorflow::DataType::DT_INT32);
    tproto->add_int_val(1);
    tproto->add_int_val(2);
    tproto->mutable_tensor_shape()->add_dim()->set_size(2);
    board.write_tensor(&writer, time_step * 20, time_step, "tensor", tproto);*/
  }

  return 0;
}
