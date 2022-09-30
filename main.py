import cv2
import yaml
import tensorflow as tf

# from pycoral.adapters.common import input_size
# from pycoral.adapters.detect import get_objects
# from pycoral.utils.dataset import read_label_file
# from pycoral.utils.edgetpu import make_interpreter
# from pycoral.utils.edgetpu import run_inference

class ObjectDetection():

    def __init__(self, MODEL_PATH):
        self.input_dims = self.__read_yaml__(MODEL_PATH)['input_size']
        self.labels = self.__read_yaml__(MODEL_PATH)['labels']
        self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH + '/model.tflite',
                                              experimental_delegates=[tf.lite.experimental.load_delegate('libedgetpu.so.1')]
                                              )
    def __read_yaml__(self, MODEL_PATH):
        return yaml.safe_load(open(MODEL_PATH + "/model_properties.yaml", 'r'))

    def run_inference(self, frame):
        pass


if __name__ == '__main__':

    #-- initialize camera
    cap = cv2.VideoCapture(0) # set device
    cap.set(4,360) #-- set height
    cap.set(3,640) #-- set width

    #-- initialize model
    MODEL_PATH = "./models/SSD_FPN_MobileNet_V1"
    inference_engine = ObjectDetection(MODEL_PATH)

    while cap.isOpened():
        ret, frame = cap.read() #-- read frame
                
        if not ret: #-- if frame is empty
            break

        cv2.imshow('frame', frame) #-- plot image

        if cv2.waitKey(1) & 0xFF == ord('q'): #-- if key: 'q' is pressed
            break
