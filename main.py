import cv2
import yaml
import tflite_runtime.interpreter as tflite
import numpy as np
from pycoral.adapters.detect import get_objects
import argparse

class ObjectDetection():

    def __init__(self, MODEL_PATH, threshold):
        self.input_dims = self.__read_yaml__(MODEL_PATH)['input_size']
        self.labels = self.__read_yaml__(MODEL_PATH)['labels']
        self.interpreter = tflite.Interpreter(model_path=MODEL_PATH + "/model.tflite",
                                              experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.interpreter.allocate_tensors()
        self.threshold = threshold 
        self.scale = np.array([self.input_dims[0]/640, self.input_dims[1]/360])

    def __read_yaml__(self, MODEL_PATH):
        return yaml.safe_load(open(MODEL_PATH + "/model_properties.yaml", 'r'))

    def preprocessImage(self, image):
        new_image = cv2.resize(image, (self.input_dims[0], self.input_dims[1]), interpolation=cv2.INTER_AREA)
        return np.expand_dims(new_image,(0))

    def run_inference(self, frame):
        self.interpreter.set_tensor(self.input_details[0]['index'], self.preprocessImage(frame))
        self.interpreter.invoke()
        return get_objects(self.interpreter, self.threshold, self.scale)

    def image_objects(self, image, objs, labels):
        for obj in objs:
            bbox = obj.bbox
            image =cv2.rectangle(image, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax),
                               color=(255,50,255), thickness=2)
            image = cv2.putText(image, '%s\n%.2f' % (labels[obj.id], obj.score),
                    (bbox.xmin + 10, bbox.ymin + 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,255))
        return image


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Coral: Object Detection Models")
    parser.add_argument("model_num", help="select model from available models (8)", type=int)
    parser.add_argument("threshold", help="display only objects if this threshold is met", type=float)
    args = parser.parse_args()

    # available models 
    models = [
              'EfficientDet-Lite0',
              'EfficientDet-Lite1',
              'EfficientDet-Lite2',
              'EfficientDet-Lite3',
              'EfficientDet-Lite3x',
              'SSD_FPN_MobileNet_V1',
              'SSD_MobileNet_V2',
              'SSDLite_MobileDet',
              ]

    #-- initialize camera
    cap = cv2.VideoCapture(0) # set device
    cap.set(4,360) #-- set height
    cap.set(3,640) #-- set width

    #-- initialize model
    MODEL_PATH = "./models/" + models[args.model_num]
    inference_engine = ObjectDetection(MODEL_PATH, args.threshold)

    while cap.isOpened():
        ret, frame = cap.read() #-- read frame
                
        if not ret: #-- if frame is empty
            break

        #-- run inference
        objects_detected = inference_engine.run_inference(frame)

        img_with_obj = inference_engine.image_objects(frame, objects_detected, inference_engine.labels) 

        cv2.imshow('frame', img_with_obj) #-- plot image

        if cv2.waitKey(1) & 0xFF == ord('q'): #-- if key: 'q' is pressed
            break
