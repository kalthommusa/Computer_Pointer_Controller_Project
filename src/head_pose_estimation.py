import cv2
import numpy as np
from openvino.inference_engine import IECore


class Head_Pose_Estimation_Model:
    '''
    The Head Pose Estimation Model Class
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

        supported_layers = self.plugin.query_network(network=self.network,device_name=self.device)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]

        if len(unsupported_layers) != 0 and self.device == 'CPU':
            print("unsupported layers found:{}".format(unsupported_layers))
            
            if not self.extensions == None:
                print("Adding cpu_extension")

                self.plugin.add_extension(self.extensions, self.device)
                
                supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
                unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
                
                if len(unsupported_layers)!=0:
                    print("Issue still exists")
                    exit(1)

                print("Issue resolved after adding extensions")
            else:
                print("provide path of cpu extension")
                exit(1)

        self.exec_net = self.plugin.load_network(network=self.network,device_name=self.device,num_requests=1)
        self.input_name = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_name].shape
        self.output_name = next(iter(self.network.outputs))
        self.output_shape = self.network.outputs[self.output_name].shape

    def predict(self, image):

        img_processed = self.preprocess_input(image.copy())
        
        outputs = self.exec_net.infer({self.input_name : img_processed})
        
        result = self.preprocess_output(outputs)
        
        return result

    def check_model(self):
        pass
        
    def preprocess_input(self, image):

        image_resized = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))

        img_processed = np.transpose(np.expand_dims(image_resized, axis=0), (0,3,1,2))
        
        return img_processed
        
    def preprocess_output(self, outputs):

        outs = []

        outs.append(outputs['angle_y_fc'].tolist()[0][0])
        outs.append(outputs['angle_p_fc'].tolist()[0][0])
        outs.append(outputs['angle_r_fc'].tolist()[0][0])

        return outs