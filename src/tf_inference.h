#include <dyna_utility.h>
#include <dyna_cv_utility.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/cc/framework/ops.h>

using namespace std;
using namespace cv;
using namespace tensorflow;


struct dati_inference {
        int id;
        string label;
        int prob;
        Rect rect;
};

class tf_inference {
 public:
   tf_inference(){}
   ~tf_inference(){}
   
   int setup(const string &modelfile, const string &labelfile, bool growth = true, float fraction = 0.5);
   vector< Tensor > inference();
   string get_label(int index);
   void set_inputs(const vector< pair<string, Tensor> > &input);
   void set_outputs(const vector< string > &input);
   Tensor mat_to_tensor(Mat &pix);
   void close();
 private:
   vector<string> load_labels(const char * file_name);
   float _threshold;
   vector<string> _tf_label_map;
   unique_ptr<Session> _tf_session;
   vector< pair<string, Tensor> > _inputs;
   vector< string > _outputs;
   };
