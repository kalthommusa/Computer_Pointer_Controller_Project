import cv2
import numpy as np
from openvino.inference_engine import IECore


class Facial_Landmarks_Detection_Model:
    '''
    The Facial Landmark Detection Model Class
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
            if not self.extensions == None:
                print("Adding cpu_extension")
                self.plugin.add_extension(self.extensions,self.device)
                
                supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
                unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
                
                if len(unsupported_layers)!=0:
                    print("Issue still exists")
                    exit(1)

                print("Issue resolved after adding extensions")
            else:
                print("provide path of cpu extension")
                exit(1)

        self.exec_net = self.plugin.load_network(network=self.network, device_name=self.device, num_requests=1)
        self.input_name = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_name].shape
        self.output_name = next(iter(self.network.outputs))
        self.output_shape = self.network.outputs[self.output_name].shape

    def predict(self, image):

        img_processed = self.preprocess_input(image.copy())
        outputs = self.exec_net.infer({self.input_name : img_processed})
        coords = self.preprocess_output(outputs) 

        h = image.shape[0]
        w = image.shape[1]
        
        coords = coords* np.array([w, h, w, h])
        coords = coords.astype(np.int32)
        
        le_xmin = coords[0]-10
        le_ymin = coords[1]-10
        le_xmax = coords[0]+10
        le_ymax = coords[1]+10
        
        re_xmin = coords[2]-10
        re_ymin = coords[3]-10
        re_xmax = coords[2]+10
        re_ymax = coords[3]+10

        left_eye = image[le_ymin:le_ymax, le_xmin:le_xmax]
        right_eye = image[re_ymin:re_ymax, re_xmin:re_xmax]
        
        eye_coordinates = [[le_xmin, le_ymin, le_xmax, le_ymax], [re_xmin, re_ymin, re_xmax, re_ymax]]
        
        return left_eye, right_eye, eye_coordinates

    def check_model(self):
        pass
        
    def preprocess_input(self, image):

        image_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_cvt, (self.input_shape[3], self.input_shape[2]))
        img_processed = np.transpose(np.expand_dims(image_resized, axis=0), (0, 3, 1, 2))

        return img_processed
        
    def preprocess_output(self, outputs):

        outs = outputs[self.output_name][0]
        left_eye_x = outs[0].tolist()[0][0]
        left_eye_y = outs[1].tolist()[0][0]
        right_eye_x = outs[2].tolist()[0][0]
        right_eye_y = outs[3].tolist()[0][0]
        
        return (left_eye_x, left_eye_y, right_eye_x, right_eye_y)