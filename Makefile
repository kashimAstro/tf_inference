PWD_DYNA=${LIBRERIA_DYNA}
TARGET=${TARGET_DYNA}

ifeq ($(TARGET),)
$(error la variabile di environment TARGET_DYNA non esiste, segliere: [x86_64|arm-allwinner|arm-broadcom])
endif
ifeq ($(PWD_DYNA),)
$(error la variabile di environment LIBRERIA_DYNA non esiste, segliere: [il path di lavoro assoluto])
endif

$(info Compilazione con target: ${TARGET} in folder: ${PWD_DYNA})

include ../../makefile/${TARGET}

CCINC=-I${PWD_DYNA}/include/opencv2 -I${PWD_DYNA}/include -I/usr/include/ ${EXTERNAL_INCLUDE} -I${PWD_DYNA}/include/tensorflow

DEFINE=

ifeq ($(TARGET), cuda9)
        DEFINE+=-DGPU -DCUDNN
endif
ifeq ($(TARGET), cuda8)
        DEFINE+=-DGPU -DCUDNN
endif

LIBEXT=${EXTERNAL_LIB} -L${PWD_DYNA}/lib/${TARGET}/ -ldyna_yolo -ldyna_cv_utility -ldyna_utility -ldarknet -lopencv_dnn -lopencv_ml -lopencv_shape -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_photo -lopencv_imgcodecs -lopencv_video -lopencv_objdetect -lopencv_imgproc -lopencv_flann -lopencv_core -littnotify -llibprotobuf -llibwebp -llibjasper -lippiw -lpthread -lX11 -lpng -fopenmp -lgtk-3 -lgdk-3 -lpangocairo-1.0 -lpango-1.0 -latk-1.0 -lcairo-gobject -lcairo -lgdk_pixbuf-2.0 -lgio-2.0 -lgthread-2.0 -lpng -lz -lIlmImf -lgstbase-1.0 -lgstreamer-1.0 -lgobject-2.0 -lglib-2.0 -lgstvideo-1.0 -lgstapp-1.0 -lgstriff-1.0 -lgstpbutils-1.0 -lavcodec -lavformat -lavutil -lswscale -ldl -lm -lpthread -lrt -ltiff -lImath -lIlmImf -lIex -lHalf -lIlmThread -ljpeg -lpng -lippicv -ldc1394 -ltbb -ltensorflow_framework -ltensorflow_cc -ltensorflow_lite -lcrypto 

CFLAGS=-std=c++11 ${DEFINE}

CC=g++

TFMAIN=$(shell basename $(CURDIR))
TFLMAIN=$(shell basename $(CURDIR))-lite

tf_main.o: tf_main.cpp
	$(CC) $(CFLAGS) $(CCINC) -c tf_main.cpp

tflite_main.o: tflite_main.cpp
	$(CC) $(CFLAGS) $(CCINC) -c tflite_main.cpp

tf_inference.o: tf_inference.cpp
	$(CC) $(CFLAGS) $(CCINC) -c tf_inference.cpp

tflite_inference.o: tflite_inference.cpp
	$(CC) $(CFLAGS) $(CCINC) -c tflite_inference.cpp

$(TFMAIN): tf_inference.o tf_main.o
	$(CC) tf_inference.o tf_main.o $(LIBEXT) -o $(TFMAIN)
	
$(TFLMAIN): tflite_inference.o tflite_main.o
	$(CC) tflite_inference.o tflite_main.o $(LIBEXT) -o $(TFLMAIN)

clean:
	rm -f *.o $(MAIN)
