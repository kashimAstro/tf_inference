#include <iostream>
#include "tf_inference.h"

int main(int argc, const char * argv[]) {
    if(argc != 4) {
	cerr << "Usage: image frozen-model labels\n";
	return 0;
    }
    string imgfile   = argv[1];
    string modelfile = argv[2];
    string labelfile = argv[3];

    int threshold = 0.2;
    Mat image = imread(imgfile.c_str(), IMREAD_COLOR);
    //resize for network input

    tf_inference tf_inf;
    if( tf_inf.setup(modelfile, labelfile) != -1 ) {
       Tensor intensor = tf_inf.mat_to_tensor(image);
       tf_inf.set_inputs({{"image_tensor:0", intensor}});
       tf_inf.set_outputs({"detection_boxes:0", "detection_scores:0", "detection_classes:0", "num_detections:0"});
       vector< Tensor > dati = tf_inf.inference();

       TTypes<float>::Flat scores         = dati[1].flat<float>();
       TTypes<float>::Flat classes        = dati[2].flat<float>();
       TTypes<float>::Flat num_detections = dati[3].flat<float>();
       auto boxes                         = dati[0].flat_outer_dims<float,3>();

       int detections_count = (int)(num_detections(0));
       for(int i = 0; i < detections_count; i++) {
          if(scores(i) > threshold) {
             float box_class = classes(i);
             int x1 = float(image.size().width)  * boxes(0,i,1);
             int y1 = float(image.size().height) * boxes(0,i,0);
             int x2 = float(image.size().width)  * boxes(0,i,3);
             int y2 = float(image.size().height) * boxes(0,i,2);
             Point tl = Point((int)x1, (int)y1);
             Point br = Point((int)x2, (int)y2);
             string msg = tf_inf.get_label(box_class)+" "+to_string((scores(i) * 100))+"%";
	     Rect rect = Rect(tl,br);
             rectangle(image, Rect(rect.x,rect.y-10,rect.width,10), Scalar(0,0,0), FILLED);
             putText(image, msg.c_str(), Point(rect.x,rect.y-5), FONT_HERSHEY_DUPLEX, 0.25, Scalar(255,255,255));
             rectangle(image, rect, Scalar(0,0,255));
         }
      }
      imshow("output", image);
      waitKey(0);
    }
    return EXIT_SUCCESS;
}

