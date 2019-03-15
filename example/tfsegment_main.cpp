#include <iostream>
#include "tf_inference.h"

void move_contour(vector<Point>& contour, int dx, int dy){
    for (size_t i=0; i<contour.size(); i++) {
        contour[i].x += dx;
        contour[i].y += dy;
    }
}

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
      tf_inf.set_outputs({"detection_boxes:0", "detection_scores:0", "detection_classes:0", "detection_masks:0", "num_detections:0"});
      vector< Tensor > dati = tf_inf.inference();
      auto boxes                                     = dati[0].flat_outer_dims<float,3>();
      tensorflow::TTypes<float>::Flat scores         = dati[1].flat<float>();
      tensorflow::TTypes<float>::Flat classes        = dati[2].flat<float>();
      auto masks                                     = dati[3];
      tensorflow::TTypes<float>::Flat num_detections = dati[4].flat<float>();
      
      cerr << "BOXES= " << dati[0].DebugString() << endl;
      cerr << "**********************" << endl;
      cerr << "SCORE= " << dati[1].DebugString() << endl;
      cerr << "**********************" << endl;
      cerr << "CLASS= " << dati[2].DebugString() << endl;
      cerr << "**********************" << endl;
      cerr << "MASKS= " << dati[3].DebugString() << endl;
      cerr << "**********************" << endl;
      cerr << "NUMDE= " << dati[3].DebugString() << endl;
      cerr << "**********************" << endl;
      
      cerr << "Mask output TensorShape ["
           << masks.shape().dim_size(0) << ", "
           << masks.shape().dim_size(1) << ", "
           << masks.shape().dim_size(2) << ", "
           << masks.shape().dim_size(3) << "]" << endl;

      RNG rng(12345);
      int detections_count = (int)(num_detections(0));
      for(int i = 0; i < detections_count; i++) {
         if(scores(i) > threshold) {
            float box_class = classes(i);
            
            int x1 = float(image.size().width)  * boxes(0,i,1);
            int y1 = float(image.size().height) * boxes(0,i,0);
            int x2 = float(image.size().width)  * boxes(0,i,3);
            int y2 = float(image.size().height) * boxes(0,i,2);
            
            Point tl   = Point((int)x1, (int)y1);
            Point br   = Point((int)x2, (int)y2);
            Rect rect  = Rect(tl,br);
            
            string msg = tf_inf.get_label(box_class)+" "+to_string((scores(i) * 100))+"%";
            rectangle(image, Rect(rect.x,rect.y-10,rect.width,10), Scalar(0,0,0), FILLED);
            putText(image, msg.c_str(), Point(rect.x,rect.y-5), FONT_HERSHEY_DUPLEX, 0.25, Scalar(255,255,255));
	    rectangle(image, rect, Scalar(0,0,255));
   
            int h_s = masks.shape().dim_size(2);
            int w_s = masks.shape().dim_size(3);
            if (h_s != 0 && w_s != 0 && i == 0) {
               try {
                  cv::Mat result_seg(masks.dim_size(2), masks.dim_size(3), CV_32FC1, masks.flat<float>().data());
                  resize(result_seg,result_seg,Size(rect.width,rect.height));
                  
		  result_seg.convertTo(result_seg,CV_8U,1.0*255);
		  vector<vector<Point>> cnts;
                  vector<Vec4i> hierarchy;
   		  findContours(result_seg, cnts, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

		  for(unsigned int i = 0; i< cnts.size(); i++) {
		    Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
		    move_contour( cnts[i], rect.x, rect.y);
		    drawContours( image, cnts, (int)i, color, 2, LINE_8, hierarchy, 0 );
		    }
		  string n1 = "map_segment_"+to_string(i)+".png";
		  string n2 = "segment_"+to_string(i)+".png";
		  imwrite(n1,result_seg);
		  imwrite(n2,image); 
                  }
               catch(Exception &e){
                  cerr << e.what() << endl;
                  }
               }
            }
         }
      imshow("output",image);
      waitKey(0);
      }
   return EXIT_SUCCESS;
   }
