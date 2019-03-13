#include <iostream>
#include "tflite_inference.h"

int main(int argc, const char * argv[]) {
    if(argc != 4) {
	cerr << "Usage: image frozen-model labels\n";
	return 0;
    }
    string imgfile   = argv[1];
    string modelfile = argv[2];
    string labelfile = argv[3];

    Mat image = imread(imgfile.c_str(), IMREAD_COLOR);
    //resize for network input

    tflite_inference tflite_inf;
    if( tflite_inf.setup(modelfile, labelfile) != -1 ) {
       vector< dati_inference > dati = tflite_inf.inference(image,0.01);
       for(unsigned int i = 0, y = 1; i < dati.size(); i++, y++) {
	  string msg = dati[i].label+" "+to_string(dati[i].prob);
          cerr << msg.c_str() << endl;
	  putText(image, msg.c_str(), Point(10, y*20), FONT_HERSHEY_DUPLEX, 0.55, Scalar(55,55,55));
       }
       imshow("output", image);
       waitKey(0);
    }
    return EXIT_SUCCESS;
}

