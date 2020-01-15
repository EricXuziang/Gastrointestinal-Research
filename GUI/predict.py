

'''
    Author : Ziang Xu
    Student number : 180104048
    Code : Bulid classification, segmentation method and Grad-CAM visualization.

'''

import tensorflow as tf
import keras
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras.utils.np_utils import *
import matplotlib.cm as cm
from vis.visualization import visualize_cam
from vis.utils import utils
from keras import activations
from matplotlib import pyplot as plt
from vis.visualization import visualize_saliency, overlay
from keras.preprocessing.image import ImageDataGenerator, array_to_img, load_img
from keras import backend as K
from keras.regularizers import l2
# Custom loss functions, in order to load model weights.
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    
def generalized_dice_coeff(y_true, y_pred):
    Ncl = y_pred.shape[-1]
    w = K.zeros(shape=(Ncl,))
    w = K.sum(y_true, axis=(0,1,2))
    w = 1/(w**2+0.000001)
    # Compute gen dice coef:
    numerator = y_true*y_pred
    numerator = w*K.sum(numerator,(0,1,2,3))
    numerator = K.sum(numerator)
    denominator = y_true+y_pred
    denominator = w*K.sum(denominator,(0,1,2,3))
    denominator = K.sum(denominator)
    gen_dice_coef = 2*numerator/denominator
    return gen_dice_coef

def generalized_dice_loss(y_true, y_pred):
    return 1 - generalized_dice_coeff(y_true, y_pred)

class Vgg16Predictor:
    def __init__(self, model_path):
        print("load model")
        self.model=load_model('model/model1.h5')
    # Images prediction.    
    def predict(self, img):
        
        img=img.resize((224,224))
        self.x=image.img_to_array(img)
        self.x=np.expand_dims(self.x,axis=0)
        self.y=self.model.predict(self.x,verbose=1)
        self.y=self.y.argmax(axis=-1)
        self.label=['dyed-lifted-polyps','dyed-resection-margins','esophagitis','normal-cecum',
               'normal-pylorus','normal-z-line','polyps','ulcerative-colitis']
        self.label=np.array(self.label)

        return (self.label[self.y])
        
    
    def heatmap(self, path):
        # Grad-CAM requires the category of the image as input. So we need predict image to get the label first.
        self.img=utils.load_img(path,target_size=(224,224))
        self.x=image.img_to_array(self.img)
        self.x=np.expand_dims(self.x,axis=0)
        self.y=self.model.predict(self.x,verbose=1)
        self.y=self.y.argmax(axis=-1)
        self.label=['dyed-lifted-polyps','dyed-resection-margins','esophagitis','normal-cecum',
                    'normal-pylorus','normal-z-line','polyps','ulcerative-colitis']
        self.label=np.array(self.label)
        self.layer_idx = utils.find_layer_idx(self.model, 'dense_1')

        # Swap softmax with linear
        self.model.layers[self.layer_idx].activation = activations.linear
        self.model = utils.apply_modifications(self.model)
       
        for modifier in ['guided']:
            
            f, ax = plt.subplots(1,1)
            plt.suptitle(self.label[self.y])
            
            # Model, layer id, class ,image as input.
            self.grads = visualize_cam(self.model, self.layer_idx, filter_indices=self.y, 
                                       seed_input=self.img, backprop_modifier=modifier)        
            # Lets overlay the heatmap onto original image.    
            self.jet_heatmap = np.uint8(cm.jet(self.grads)[..., :3] * 255)
            self.a=ax.imshow(overlay(self.jet_heatmap, self.img))
            
            plt.savefig("image/attention.jpg")
            return self.a
    # Load U-net++ model and segement images.
    def segmentation(self,path):
        self.path='model/unet.h5'
        self.model1=load_model(self.path ,custom_objects={'generalized_dice_loss': generalized_dice_loss,'mean_iou':mean_iou,'dice_coef':dice_coef})
        self.img=utils.load_img(path,target_size=(224,224))
        self.img=image.img_to_array(self.img)
        self.img=np.expand_dims(self.img,axis=0)
        self.img = self.img.astype('float32')
        self.img /= 255.
        self.img_mask=self.model1.predict(self.img,verbose=1)
        self.imgs=self.img_mask
        for i in range(self.imgs.shape[0]):
			
            self.img = self.imgs[i]
            self.img = array_to_img(self.img)
            self.img.save("image/mask.jpg")

        


if __name__ == "__main__":
    pass
    # m = Vgg16Predictor('model/model1.h5')
    # m.segmentation("2.jpg")
