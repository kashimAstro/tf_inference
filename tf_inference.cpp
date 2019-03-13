#include "tf_inference.h"

int tf_inference::setup(const string &modelfile, const string &labelfile, bool growth, float fraction) {
   if(modelfile == "" || labelfile == ""){
      cerr << "error: load model | labels" << endl; 
      return -1;
      }
   _tf_label_map = load_labels(labelfile.c_str());
   if(_tf_label_map.empty()){
      cerr << "error: load labels vector" << endl;
      return -1;
      }
   
   tensorflow::GraphDef graph_def;
   tensorflow::Status load_graph_status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), modelfile.c_str(), &graph_def);
   if (!load_graph_status.ok()) {
      cerr << "error: open graph" << endl;
      return -1;
      }
   tensorflow::SessionOptions session_options;
   session_options.config.mutable_gpu_options()->set_allow_growth(growth);
   session_options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(fraction);

   _tf_session.reset(tensorflow::NewSession(session_options));
   if( _tf_session->Create(graph_def).ok() )
	return 0;
   return -1;
   }

string tf_inference::get_label(int index) {
	return _tf_label_map[index];
}

void tf_inference::set_inputs(const vector< pair<string, Tensor> > &input) {
   _inputs = input;
}

void tf_inference::set_outputs(const vector< string > &output) {
   _outputs = output;
}

Tensor tf_inference::mat_to_tensor(Mat &pix) {
   int input_height = pix.size().height;
   int input_width  = pix.size().width;

   tensorflow::Tensor img_tf_with_shared_data(tensorflow::DT_UINT8, {1, input_height, input_width, pix.channels()});
   uint8_t *pdata = img_tf_with_shared_data.flat<uint8_t>().data();
   Mat data_img(input_height, input_width, CV_8UC3, pdata);
   pix.convertTo(data_img, CV_8UC3);
   return img_tf_with_shared_data;
}

vector< Tensor > tf_inference::inference() {
   //vector< dati_inference > dati;
   /*if(pix.empty())
      return dati;

   int input_height = pix.size().height;
   int input_width  = pix.size().width;

   tensorflow::Tensor img_tf_with_shared_data(tensorflow::DT_UINT8, {1, input_height, input_width, pix.channels()});
   uint8_t *pdata = img_tf_with_shared_data.flat<uint8_t>().data();
   Mat data_img(input_height, input_width, CV_8UC3, pdata);
   pix.convertTo(data_img, CV_8UC3);*/

   vector<Tensor> outputs;
   /*Status run_status = _tf_session->Run({{"image_tensor:0", img_tf_with_shared_data}}, 
				        {"detection_boxes:0", 
                                         "detection_scores:0", 
                                         "detection_classes:0", 
					 "num_detections:0"}, {}, &outputs);*/

   Status run_status = _tf_session->Run(_inputs, _outputs, {}, &outputs);
   if(!run_status.ok()) {
      cerr << "session->run= " << run_status << endl;
      return outputs;
      }

/* tensorflow::TTypes<float>::Flat scores = outputs[1].flat<float>();
   tensorflow::TTypes<float>::Flat classes = outputs[2].flat<float>();
   tensorflow::TTypes<float>::Flat num_detections = outputs[3].flat<float>();
   auto boxes = outputs[0].flat_outer_dims<float,3>();
   
   int detections_count = (int)(num_detections(0));
   for(int i = 0; i < detections_count; i++) {
      if(scores(i) > threshold) {
         float box_class = classes(i);
         
         int x1 = float(input_width)  * boxes(0,i,1);
         int y1 = float(input_height) * boxes(0,i,0);
         int x2 = float(input_width)  * boxes(0,i,3);
         int y2 = float(input_height) * boxes(0,i,2);

	 Point tl = Point((int)x1, (int)y1);
         Point br = Point((int)x2, (int)y2);

         dati_inference dnet;
         dnet.id    = box_class; 
         dnet.label = _tf_label_map[box_class];
         dnet.prob  = (scores(i)  * 100);
         dnet.rect  = Rect(tl,br);
         dati.push_back(dnet);
         }
      }*/
   return outputs;
   }

void tf_inference::close() {
   }

vector<string> tf_inference::load_labels(const char * file_name) {
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

