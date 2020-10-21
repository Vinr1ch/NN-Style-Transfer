import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import time
#from keras.applications import vgg19
#from keras import backend as K
import random
from scipy.optimize import fmin_l_bfgs_b, minimize
from keras.preprocessing.image import load_img, save_img, img_to_array
#from scipy.misc import imsave, imresize
#from scipy.optimize import fmin_l_bfgs_b   # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
import warnings

random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)
tf.compat.v1.disable_eager_execution()
#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


CONTENT_IMG_PATH =  "fb.jpg"      #TODO: Add this.
STYLE_IMG_PATH = "palace.jpg"          #TODO: Add this.


CONTENT_IMG_H = 500
CONTENT_IMG_W = 500

STYLE_IMG_H = 500
STYLE_IMG_W = 500

CONTENT_WEIGHT = 0.1    # Alpha weight.
STYLE_WEIGHT = 1.0      # Beta weight.
TOTAL_WEIGHT = 1.0

TRANSFER_ROUNDS = 3

class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

#=============================<Helper Fuctions>=================================
'''
TODO: implement this.
This function should take the tensor and re-convert it to an image.
'''
def deprocessImage(img):
    return img


def gramMatrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram



#========================<Loss Function Builder Functions>======================

def eval_loss_and_grads(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((1, 3, CONTENT_IMG_H, CONTENT_IMG_W))
    else:
        x = x.reshape((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
		# f_outputs is defined below
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

def styleLoss(style, gen):
    assert K.ndim(style) == 3
    assert K.ndim(gen) == 3
    S = gram_matrix(style)
    C = gram_matrix(gen)
    channels = 3
    size = CONTENT_IMG_H * CONTENT_IMG_W
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


def contentLoss(content, gen):
    return K.sum(K.square(gen - content))


def totalLoss(x):
    assert K.ndim(x) == 4
    if K.image_data_format() == 'channels_first':
        a = K.square(x[:, :, :CONTENT_IMG_H - 1, :CONTENT_IMG_W - 1] - x[:, :, 1:, :CONTENT_IMG_W - 1])
        b = K.square(x[:, :, :CONTENT_IMG_H - 1, :CONTENT_IMG_W - 1] - x[:, :, :CONTENT_IMG_H - 1, 1:])
    else:
        a = K.square(x[:, :CONTENT_IMG_H - 1, :CONTENT_IMG_W - 1, :] - x[:, 1:, :CONTENT_IMG_W - 1, :])
        b = K.square(x[:, :CONTENT_IMG_H - 1, :CONTENT_IMG_W - 1, :] - x[:, :CONTENT_IMG_H - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

def gram_matrix(x):
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram



#=========================<Pipeline Functions>==================================
'''
def getRawData():
    print("   Loading images.")
    print("      Content image URL:  \"%s\"." % CONTENT_IMG_PATH)
    print("      Style image URL:    \"%s\"." % STYLE_IMG_PATH)
    cImg = load_img(CONTENT_IMG_PATH)
    tImg = cImg.copy()
    sImg = load_img(STYLE_IMG_PATH)
    print("      Images have been loaded.")
    return ((cImg, CONTENT_IMG_H, CONTENT_IMG_W), (sImg, STYLE_IMG_H, STYLE_IMG_W), (tImg, CONTENT_IMG_H, CONTENT_IMG_W))
'''


def preprocessData(image_path):
    img = load_img(image_path, target_size=(CONTENT_IMG_H, CONTENT_IMG_W))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

# util function to convert a tensor into a valid image
def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, CONTENT_IMG_H, CONTENT_IMG_W))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((CONTENT_IMG_H, CONTENT_IMG_W, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


'''
TODO: Allot of stuff needs to be implemented in this function.
First, make sure the model is set up properly.
Then construct the loss function (from content and style loss).
Gradient functions will also need to be created, or you can use K.Gradients().
Finally, do the style transfer with gradient descent.
Save the newly generated and deprocessed images.
'''
def styleTransfer(cData, sData, tData):
    print("   Building transfer model.")
    contentTensor = K.variable(cData)
    styleTensor = K.variable(sData)
    genTensor = K.placeholder((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
    #inputTensor = K.concatenate([contentTensor, styleTensor, genTensor], axis=0)
    inputTensor = K.concatenate([cData,
                              sData,
                              tData], axis=0)

    model = vgg19.VGG19(input_tensor=inputTensor, weights='imagenet', include_top=False) #implement

    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])


    print("   VGG19 model loaded.")
    loss = 0.0
    styleLayerNames = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
    contentLayerName = "block5_conv2"


    print("   Calculating content loss.")
    contentLayer = outputs_dict[contentLayerName]
    contentOutput = contentLayer[0, :, :, :]
    genOutput = contentLayer[2, :, :, :]
    layer_features = outputs_dict['block5_conv2']
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss += CONTENT_WEIGHT * contentLoss(base_image_features, combination_features)

                                      # layers
    feature_layers = ['block1_conv1', 'block2_conv1',
                      'block3_conv1', 'block4_conv1',
                      'block5_conv1']
    print("   Calculating style loss.")
    for layer_name in feature_layers:
        layer_features = outputs_dict[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = styleLoss(style_reference_features, combination_features)
        loss += (STYLE_WEIGHT / len(feature_layers)) * sl

    loss += TOTAL_WEIGHT * totalLoss(tData)

    # get the gradients of the generated image wrt the loss
    grads = K.gradients(loss, tData)

    outputs = [loss]
    if isinstance(grads, (list, tuple)):
        outputs += grads
    else:
        outputs.append(grads)

    f_outputs = K.function([tData], outputs)
    print(f_out)

    x = preprocessData(CONTENT_IMG_PATH)
    # TODO: Setup gradients or use K.gradients().
    print("   Beginning transfer.")
    for i in range(TRANSFER_ROUNDS):
        print('Start of iteration', i)
        start_time = time.time()

        # fmin_l_bfgs_b
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss,
                                         x.flatten(),
                                         fprime=evaluator.grads,
                                         maxfun=50)

        print('Current loss value:', min_val)
        # save current generated image
        img = deprocess_image(x.copy())
        fname = result_prefix + '_at_iteration_%d.png' % i
        save_img(fname, img)
        end_time = time.time()
        print('Image saved as', fname)
        print('Iteration %d completed in %ds' % (i, end_time - start_time))
    print("   Transfer complete.")





#=========================<Main>================================================

def main():
    print("Starting style transfer program.")
    #raw = getRawData()
    cData = preprocessData(CONTENT_IMG_PATH)   # Content image.
    sData = preprocessData(STYLE_IMG_PATH)   # Style image.
    if K.image_data_format() == 'channels_first':
        tData = K.placeholder((1, 3, CONTENT_IMG_H, CONTENT_IMG_W))
    else:
        tData = K.placeholder((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
    styleTransfer(cData, sData, tData)
    print("Done. Goodbye.")



if __name__ == "__main__":
    main()
