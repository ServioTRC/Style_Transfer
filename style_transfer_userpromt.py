""" An implementation of the paper "A Neural Algorithm of Artistic Style"
by Gatys et al. in TensorFlow, based upon the assignment for the CS 20SI:
"TensorFlow for Deep Learning Research" created by Chip Huyen
(huyenn@stanford.edu).

For more details related to the convolution network implementation,
please read the assignment handout:
http://web.stanford.edu/class/cs20si/assignments/a2.pdf
"""
import time

import numpy as np
import tensorflow as tf

import vgg_model
import utils
import data_input

#Avoids innecesary warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Parameters to manage experiments
STYLE = None
CONTENT = None
STYLE_IMAGE = None
CONTENT_IMAGE = None
IMAGE_HEIGHT = None
IMAGE_WIDTH = None
#Percentage of weight of the noise for intermixing with the content image
NOISE_RATIO = None

# Layers used for style features. You can change this.
STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
W = [0.5, 1.0, 1.5, 3.0, 4.0] # give more weights to deeper layers.

# Layer used for content features. You can change this.
CONTENT_LAYER = 'conv4_2'

ITERS = 10
# Learning rate of the algorithm, the smaller more precise but uses more resources for achieving it
LR = 2.0

#Weights to determine how much of the content/style image is used
CONTENT_WEIGHT = None
STYLE_WEIGHT = None

MEAN_PIXELS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
""" MEAN_PIXELS is defined according to description on their github:
https://gist.github.com/ksimonyan/211839e770f7b538e2d8
'In the paper, the model is denoted as the configuration D trained with scale jittering. 
The input images should be zero-centered by mean pixel (rather than mean image) subtraction. 
Namely, the following BGR values should be subtracted: [103.939, 116.779, 123.68].'
"""

# VGG-19 parameters file
VGG_DOWNLOAD_LINK = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat'
VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'
EXPECTED_BYTES = 534904783


def _setting_parameters():
    
    global STYLE, CONTENT, IMAGE_HEIGHT, IMAGE_WIDTH, NOISE_RATIO, ITERS, \
           CONTENT_WEIGHT, STYLE_WEIGHT, STYLE_IMAGE, CONTENT_IMAGE
    try:
        STYLE = data_input._choose_files("style")
        CONTENT = data_input._choose_files("content")
        IMAGE_HEIGHT, IMAGE_WIDTH = data_input._set_image_size()
        NOISE_RATIO = data_input._set_noise_ratio()
        ITERS = data_input._set_iterations()
        CONTENT_WEIGHT, STYLE_WEIGHT = data_input._set_weights()
        STYLE_IMAGE = 'styles/' + STYLE + '.jpg'
        CONTENT_IMAGE = 'content/' + CONTENT + '.jpg'
        
    except Exception as e:
            if e.args[0] == "CANCEL":
                print("Operation canceled")
            else:
                print(e)

def _create_content_loss(p, f):
    """ Calculate the loss between the feature representation of the
    content image and the generated image.
    
    Inputs: 
        p, f are just P, F in the paper 
        (read the assignment handout if you're confused)
        Note: we won't use the coefficient 0.5 as defined in the paper
        but the coefficient as defined in the assignment handout.
    Output:
        the content loss

    """
    return tf.reduce_sum((f - p) ** 2) / (4.0 * p.size)

def _gram_matrix(F, N, M):
    """ Create and return the gram matrix for tensor F
        Gram matrix explanation: https://en.wikipedia.org/wiki/Gramian_matrix
    """
    F = tf.reshape(F, (M, N))
    return tf.matmul(tf.transpose(F), F)

def _single_style_loss(a, g):
    """ Calculate the style loss at a certain layer
    Inputs:
        a is the feature representation of the real image
        g is the feature representation of the generated image
    Output:
        the style loss at a certain layer (which is E_l in the paper)
    """
    N = a.shape[3]  # number of filters
    M = a.shape[1] * a.shape[2]  # height times width of the feature map
    A = _gram_matrix(a, N, M)
    G = _gram_matrix(g, N, M)
    return tf.reduce_sum((G - A) ** 2 / ((2 * N * M) ** 2))

def _create_style_loss(A, model):
    """ Return the total style loss
    """
    n_layers = len(STYLE_LAYERS)
    E = [_single_style_loss(A[i], model[STYLE_LAYERS[i]]) for i in range(n_layers)]
    
    return sum([W[i] * E[i] for i in range(n_layers)])


def _create_losses(model, input_image, content_image, style_image):
    with tf.variable_scope('loss') as scope:
        with tf.Session() as sess:
            sess.run(input_image.assign(content_image)) # assign content image to the input variable
            p = sess.run(model[CONTENT_LAYER])
        content_loss = _create_content_loss(p, model[CONTENT_LAYER])

        with tf.Session() as sess:
            sess.run(input_image.assign(style_image))
            A = sess.run([model[layer_name] for layer_name in STYLE_LAYERS])                              
        style_loss = _create_style_loss(A, model)

        # Create total loss. 
        total_loss = CONTENT_WEIGHT * content_loss + STYLE_WEIGHT * style_loss

    return content_loss, style_loss, total_loss

def _create_summary(model):
    """ Create summary ops necessary
    """
    with tf.name_scope('summaries'):
        tf.summary.scalar('content loss', model['content_loss'])
        tf.summary.scalar('style loss', model['style_loss'])
        tf.summary.scalar('total loss', model['total_loss'])
        tf.summary.histogram('histogram content loss', model['content_loss'])
        tf.summary.histogram('histogram style loss', model['style_loss'])
        tf.summary.histogram('histogram total loss', model['total_loss'])
        return tf.summary.merge_all()

def train(model, generated_image, initial_image):
    """ Train your model."""
    with tf.Session() as sess:
        saver = tf.train.Saver()
        #Initialize variables
        sess.run(tf.global_variables_initializer())        
        sess.run(generated_image.assign(initial_image))
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpointsTransfer/checkpoint/'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        initial_step = model['global_step'].eval()
        
        start_time = time.time()

        skip_step = 1
        for index in range(initial_step, ITERS):
            if index >= 5 and index < 20:
                skip_step = 10
            elif index >= 20:
                skip_step = 20
            
            sess.run(model['optimizer'])
            if (index + 1) % skip_step == 0:
                #Generated image and loss
                gen_image, total_loss, summary = sess.run([generated_image, model['total_loss'],
                                                           model['summary_op']])
                elapsed_time = time.time() - start_time
                gen_image = gen_image + MEAN_PIXELS
                print('Step {}\n   Sum: {:5.1f}'.format(index + 1, np.sum(gen_image)))
                print('   Loss: {:5.1f}'.format(total_loss))
                print('   Time: {}'.format(elapsed_time))

                utils.eta(index, skip_step, elapsed_time, ITERS)
                
                start_time = time.time()
                filename = 'outputs/%d.png' % (index)
                utils.save_image(filename, gen_image)

                if (index + 1) % 20 == 0:
                    saver.save(sess, 'checkpointsTransfer/style_transfer/', index)

def main():
    _setting_parameters()
    with tf.device('/cpu:0'):   #Change to '/gpu:0' if having a GPU
        with tf.variable_scope('input') as scope:
            # use variable instead of placeholder because we're training the initial image to make it
            # look like both the content image and the style image
            input_image = tf.Variable(np.zeros([1, IMAGE_HEIGHT, IMAGE_WIDTH, 3]), dtype=tf.float32)

        #Downloads the pretrained model
        utils.download(VGG_DOWNLOAD_LINK, VGG_MODEL, EXPECTED_BYTES)
        model = vgg_model.load_vgg(VGG_MODEL, input_image)
        model['global_step'] = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        content_image = utils.get_resized_image(CONTENT_IMAGE, IMAGE_HEIGHT, IMAGE_WIDTH)
        content_image = content_image - MEAN_PIXELS
        style_image = utils.get_resized_image(STYLE_IMAGE, IMAGE_HEIGHT, IMAGE_WIDTH)
        style_image = style_image - MEAN_PIXELS

        model['content_loss'], model['style_loss'], model['total_loss'] = _create_losses(model,
                                                        input_image, content_image, style_image)

        #Optimizer looking to reduce the total_loss of the model
        #TensorFlow optimizers: https://www.tensorflow.org/api_docs/python/tf/train
        model['optimizer'] = tf.train.AdamOptimizer(LR).minimize(model['total_loss'],
                                                                 global_step=model['global_step'])
        model['summary_op'] = _create_summary(model)

        initial_image = utils.generate_noise_image(content_image, IMAGE_HEIGHT, IMAGE_WIDTH, NOISE_RATIO)
        train(model, input_image, initial_image)
    #Showing the image one it is finished
    image_path = 'outputs/%d.png' % (ITERS-1)
    utils.show_image(image_path)

if __name__ == '__main__':
    main()
    
