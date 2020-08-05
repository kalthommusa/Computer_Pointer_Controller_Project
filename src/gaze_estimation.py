import cv2
import numpy as np
from openvino.inference_engine import IECore
import math


class Gaze_Estimation_Model:
    '''
    The Gaze Estimation Model Class
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        self.model_structure = model_name
        self.model_weights = self.model_name.split('.')[0]+'.bin'
        self.plugin = None
        self.network = None
        self.exec_net = None
        self.input_name = None
        self.input_shape = None
        self.output_name = None
        self.output_shape = None


    def load_model(self):

        self.plugin = IECore()
        self.network = self.plugin.read_network(model=self.model_structure, weights=self.model_weights)

        supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]

        if len(unsupported_layers)!=0 and self.device=='CPU':
            print("unsupported layers found:{}".format(unsupported_layers))
            
            if not self.extensions==None:
                print("Adding cpu_extension")
                self.plugin.add_extension(self.extensions,self.device)
                
                supported_layers = self.plugin.query_network(network=self.network,device_name=self.device)
                unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
                
                if len(unsupported_layers)!=0:
                    print("Issue still exists")
                    exit(1)

                print("Issue resolved after adding extensions")
            else:
                print("provide path of cpu extension")
                exit(1)

        self.exec_net = self.plugin.load_network(network=self.network, device_name=self.device, num_requests=1)
        self.input_name = [i for i in self.network.inputs.keys()]
        self.input_shape = self.network.inputs[self.input_name[1]].shape
        self.output_name = [i for i in self.network.outputs.keys()]

    def predict(self, left_eye_image, right_eye_image, head_pose_angle):

        le_img_processed, re_img_processed = self.preprocess_input(left_eye_image.copy(), right_eye_image.copy())

        outputs = self.exec_net.infer({'head_pose_angles':head_pose_angle, 'left_eye_image':le_img_processed, 'right_eye_image':re_img_processed})

        mouse_coords, gaze_vec = self.preprocess_output(outputs, head_pose_angle)

        return mouse_coords, gaze_vec

    def check_model(self):
        pass
        
    def preprocess_input(self, left_eye_image, right_eye_image):

        le_image_resized = cv2.resize(left_eye_image, (self.input_shape[3], self.input_shape[2]))
        le_img_processed = np.transpose(np.expand_dims(le_image_resized, axis=0), (0, 3, 1, 2))

        re_image_resized = cv2.resize(right_eye_image,(self.input_shape[3], self.input_shape[2]))
        re_img_processed = np.transpose(np.expand_dims(re_image_resized, axis=0), (0, 3, 1, 2))

        return le_img_processed,re_img_processed
        
    def preprocess_output(self, outputs, head_pose_angle):

        gazeVector = outputs[self.output_name[0]].tolist()[0]
        roll_value = head_pose_angle[2]
        cosine_value = math.cos(roll_value*math.pi/180.0)
        sine_value = math.sin(roll_value*math.pi/180.0)

        x_val = gazeVector[0] * cosine_value + gazeVector[1] * sine_value
        y_val = -gazeVector[0] *  sine_value+ gazeVector[1] * cosine_value
                
        return (x_val, y_val), gazeVector