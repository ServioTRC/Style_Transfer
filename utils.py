""" Utils needed for the implementation of the paper "A Neural Algorithm of Artistic Style"
by Gatys et al. in TensorFlow, based upon the assignment for the CS 20SI:
"TensorFlow for Deep Learning Research" created by Chip Huyen
(huyenn@stanford.edu).

For more details related to the convolution network implementation,
please read the assignment handout:
http://web.stanford.edu/class/cs20si/assignments/a2.pdf
"""

from PIL import Image, ImageOps
import numpy as np
import scipy.misc
import tkinter as tk
from six.moves import urllib

#Avoids innecesary warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def download(download_link, file_name, expected_bytes):
    """ Download the pretrained VGG-19 model if it's not already downloaded """
    if os.path.exists(file_name):
        print("\nDataset ready. Starting style-transfer.\n")
        return
    print("\nDownloading the VGG pre-trained model. This might take a while ...")
    file_name, _ = urllib.request.urlretrieve(download_link, file_name)
    file_stat = os.stat(file_name)
    if file_stat.st_size == expected_bytes:
        print('\nSuccessfully downloaded the file: ', file_name)
        print("Starting style-transfer.\n")   
    else:
        raise Exception('\nFile ' + file_name +
                        ' might be corrupted. You should try downloading it with a browser.')

def get_resized_image(img_path, height, width, save=True):
    image = Image.open(img_path)
    # it's because PIL is column major so you have to change place of width and height
    # this is stupid, i know
    image = ImageOps.fit(image, (width, height), Image.ANTIALIAS)
    if save:
        image_dirs = img_path.split('/')
        image_dirs[-1] = 'resized_' + image_dirs[-1]
        out_path = '/'.join(image_dirs)
        if not os.path.exists(out_path):
            image.save(out_path)
    image = np.asarray(image, np.float32)
    return np.expand_dims(image, 0)

def generate_noise_image(content_image, height, width, noise_ratio=0.6):
    noise_image = np.random.uniform(-20, 20, 
                                    (1, height, width, 3)).astype(np.float32)
    return noise_image * noise_ratio + content_image * (1 - noise_ratio)

def save_image(path, image):
    # Output should add back the mean pixels we subtracted at the beginning
    image = image[0] # the image
    image = np.clip(image, 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)

def show_image(path):
    root = tk.Tk()
    image = tk.PhotoImage(file=path)
    label = tk.Label(image=image)
    label.pack()
    root.mainloop()

def eta(index, skip_step, elapsed_time, iters):
    #Calculating the estimated time for finishing the style-transfer
    if skip_step == 1:
        eta_time = elapsed_time * (iters - index)
    elif skip_step == 10:
        eta_time = elapsed_time // 10 * (iters - index)
    elif skip_step == 20:
        eta_time = elapsed_time // 20 * (iters - index)

    print('   ETA: {} seconds\n'.format(eta_time))

    
