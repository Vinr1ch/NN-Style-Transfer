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
from IPython.display import Image, display

random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

CONTENT_IMG_PATH =  "fb.jpg"      #TODO: Add this.
STYLE_IMG_PATH = "SpiderVerse_cropped.jpg"          #TODO: Add this.

result_prefix = "FBSpiderVerse"

CONTENT_IMG_H = 500
CONTENT_IMG_W = 500

CONTENT_WEIGHT =2.5e-8   # Alpha weight.
STYLE_WEIGHT = 1e-6      # Beta weight.
TOTAL_WEIGHT = 1e-6

print("   Building transfer model.")

model = vgg19.VGG19(weights="imagenet", include_top=False)   #TODO: implement.
outIm = dict([(layer.name, layer.output) for layer in model.layers])
finder = keras.Model(inputs=model.inputs, outputs=outIm)

styleLayerNames = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1",]
# The layer to use for the content loss.
contentLayerName = "block5_conv2"
print("   VGG19 model loaded.")



#=============================<Helper Fuctions>=================================
def deprocessImage(img):
    return img


def gramMatrix(x):
    look = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(look, K.transpose(look))
    return gram



#========================<Loss Function Builder Functions>======================

def styleLoss(style, gen):
    connent = gramMatrix(gen)
    newStyle = gramMatrix(style)
    lay = 3
    size = CONTENT_IMG_H * CONTENT_IMG_W
    return tf.reduce_sum(tf.square(newStyle - connent)) / (4.0 * (lay ** 2) * (size ** 2))


def contentLoss(content, gen):
    return K.sum(K.square(gen - content))


def totalLoss(x):
    a = tf.square(x[:, : CONTENT_IMG_H - 1, : CONTENT_IMG_W - 1, :] - x[:, 1:, : CONTENT_IMG_W - 1, :])
    b = tf.square(x[:, : CONTENT_IMG_H - 1, : CONTENT_IMG_W - 1, :] - x[:, : CONTENT_IMG_H - 1, 1:, :])
    return tf.reduce_sum(tf.pow(a + b, 1.25))



def computeLoss(combination_image, base_image, style_reference_image):
    input_tensor = tf.concat([base_image, style_reference_image, combination_image], axis=0)
    features = finder(input_tensor)

    # make first loss
    loss = tf.zeros(shape=())

    # Add content loss
    lFeature = features[contentLayerName]
    base = lFeature[0, :, :, :]
    newFeatures = lFeature[2, :, :, :]
    loss = loss + CONTENT_WEIGHT * contentLoss(base, newFeatures)
    # Add style loss
    for layerName in styleLayerNames:
        lFeature = features[layerName]
        styleRef = lFeature[1, :, :, :]
        newFeatures = lFeature[2, :, :, :]
        sl = styleLoss(styleRef, newFeatures)
        loss += (STYLE_WEIGHT / len(styleLayerNames)) * sl

    # Add total variation loss
    loss += TOTAL_WEIGHT * totalLoss(combination_image)
    return loss

#used to call compute loss and gradient
def computeLossAndGradients(combination_image, base_image, style_reference_image):
    with tf.GradientTape() as tape:
        loss = computeLoss(combination_image, base_image, style_reference_image)
    grads = tape.gradient(loss, combination_image)
    return loss, grads



#=========================<Pipeline Functions>==================================
'''
removed as I loaded image in prepocessdata
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

#used to open and resize image with given format
def preprocessData(image_path):
    loading = keras.preprocessing.image.load_img(image_path, target_size=(CONTENT_IMG_H, CONTENT_IMG_W))
    loading = keras.preprocessing.image.img_to_array(loading)
    loading = np.expand_dims(loading, axis=0)
    loading = vgg19.preprocess_input(loading)
    return tf.convert_to_tensor(loading)

#used to help save image
def deprocess_image(x):
    x = x.reshape((CONTENT_IMG_H, CONTENT_IMG_W, 3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB' got from stack overflow
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x


def styleTransfer():
    print("   Beginning transfer.")
    optimizer = keras.optimizers.SGD(keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=100.0, decay_steps=100, decay_rate=0.94))

    base_image = preprocessData(CONTENT_IMG_PATH)
    style_reference_image = preprocessData(STYLE_IMG_PATH)
    combination_image = tf.Variable(preprocessData(CONTENT_IMG_PATH))

    iterations = 100
    for i in range(1, iterations + 1):
        loss, grads = computeLossAndGradients(
            combination_image, base_image, style_reference_image
        )
        optimizer.apply_gradients([(grads, combination_image)])
        if i % 100 == 0:
            print("Iteration %d: loss=%.2f" % (i, loss))
            img = deprocess_image(combination_image.numpy())
            fname = result_prefix + "_at_iteration_%d.png" % i
            keras.preprocessing.image.save_img(fname, img)
    print("   Transfer complete.")

    #final = Image(result_prefix + "_at_iteration_4000.png")
    #final.save("output.jpg")






#=========================<Main>================================================

def main():
    print("Starting style transfer program.")
    '''
    raw = getRawData()
    cData = preprocessData(CONTENT_IMG_PATH)   # Content image.
    sData = preprocessData(STYLE_IMG_PATH)
    tData = preprocessData(raw[2])   # Transfer image.
    '''
    styleTransfer()
    print("Done. Goodbye.")



if __name__ == "__main__":
    main()
