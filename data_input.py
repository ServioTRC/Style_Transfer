""" Data input tools needed for the implementation of the paper "A Neural Algorithm of Artistic Style"
by Gatys et al. in TensorFlow, based upon the assignment for the CS 20SI:
"TensorFlow for Deep Learning Research" created by Chip Huyen
(huyenn@stanford.edu).

For more details related to the convolution network implementation,
please read the assignment handout:
http://web.stanford.edu/class/cs20si/assignments/a2.pdf
"""

import os

def _choose_files(type):
    print()
    if type == "style":
        print("Choose the image for the style.\nFiles:")
        dir = "./styles"
    elif type == "content":
        print("Choose the image for the content.\nFiles:")
        dir = "./content"
    for f in os.listdir(dir):
        print(f)
    while(True):
        file_name = input("\nEnter the name of the file (only JPG's), enter CANCEL for aborting: ")
        if file_name == "CANCEL":
            raise Exception("CANCEL")
        file_dir = dir + "/" + file_name + ".jpg"
        try:
            file = open(file_dir, "r")
            file.close()
            if type == "style":
                return file_name
            elif type == "content":
                return file_name
            break
        except:
            print("File name not valid. Please re-enter the name.")

def _set_image_size():
    while(True):
        try:
            IMAGE_HEIGHT = input("\nEnter the height of the image (only positive integers greater than 0), enter CANCEL for aborting: ")
            if IMAGE_HEIGHT == "CANCEL":
                raise Exception("CANCEL")
            IMAGE_HEIGHT = int(IMAGE_HEIGHT)
            if IMAGE_HEIGHT > 0:
                break
            else:
                raise Exception("BOUNDS ERROR")
        except Exception as e:
            if e.args[0] == "CANCEL":
                raise Exception("CANCEL")
            print("Height not valid. Please re-enter the value.")
            
    while (True):
        try:
            IMAGE_WIDTH = input("\nEnter the width of the image (only positive integers greater than 0), enter CANCEL for aborting: ")
            if IMAGE_WIDTH == "CANCEL":
                raise Exception("CANCEL")
            IMAGE_WIDTH = int(IMAGE_WIDTH)
            if IMAGE_WIDTH > 0:
                break
            else:
                raise Exception("BOUNDS ERROR")
        except Exception as e:
            if e.args[0] == "CANCEL":
                raise Exception("CANCEL")
            print("Width not valid. Please re-enter the value.")
            
    return IMAGE_HEIGHT, IMAGE_WIDTH

def _set_noise_ratio():
    while(True):
        try:
            NOISE_RATIO = input("\nEnter the noise ratio (values between 0 and 1 exclusive), enter CANCEL for aborting: ")
            if NOISE_RATIO == "CANCEL":
                raise Exception("CANCEL")
            NOISE_RATIO = float(NOISE_RATIO)
            if 0 < NOISE_RATIO < 1:
                return NOISE_RATIO
            else:
                raise Exception("BOUNDS ERROR")
        except Exception as e:
            if e.args[0] == "CANCEL":
                raise Exception("CANCEL")
            print("Ratio not valid. Please re-enter value.")

def _set_iterations():
    while(True):
        try:
            ITERS = input("\nEnter the number of iterations (integer greater than 0), enter CANCEL for aborting: ")
            if ITERS == "CANCEL":
                raise Exception("CANCEL")
            ITERS = int(ITERS)
            if ITERS > 0:
                return ITERS
            else:
                raise Exception("BOUNDS ERROR")
        except Exception as e:
            if e.args[0] == "CANCEL":
                raise Exception("CANCEL")
            print("Invalid number of iterations. Please re-enter the value.")

def _set_weights():
    while (True):
        try:
            CONTENT_WEIGHT = input("\nEnter the content weight of the image (value between 0 and 1 exclusive), enter CANCEL for aborting: ")
            if CONTENT_WEIGHT == "CANCEL":
                raise Exception("CANCEL")
            CONTENT_WEIGHT = float(CONTENT_WEIGHT)
            if 0 < CONTENT_WEIGHT < 1:
                break
            else:
                raise Exception("BOUNDS ERROR")
        except Exception as e:
            if e.args[0] == "CANCEL":
                raise Exception("CANCEL")
            print("Height not valid. Please re-enter the value.")
    while (True):
        try:
            STYLE_WEIGHT = input("\nEnter the style weight of the image (value between 0 and 1 exclusive), enter CANCEL for aborting: ")
            if STYLE_WEIGHT == "CANCEL":
                raise Exception("CANCEL")
            STYLE_WEIGHT = float(STYLE_WEIGHT)
            if 0 < STYLE_WEIGHT < 1:
                break
            else:
                raise Exception("BOUNDS ERROR")
        except Exception as e:
            if e.args[0] == "CANCEL":
                raise Exception("CANCEL")
            print("Width not valid. Please re-enter the value.")
    return CONTENT_WEIGHT, STYLE_WEIGHT
