# Style Transfer Project using TensorFlow

An implementation of the paper "A Neural Algorithm of Artistic Style"
by Gatys et al. in TensorFlow, based upon the assignment for the CS 20SI:
"TensorFlow for Deep Learning Research" created by Chip Huyen
(huyenn@stanford.edu).

For more details related to the convolution network implementation,
please read the assignment handout:
http://web.stanford.edu/class/cs20si/assignments/a2.pdf

## Details of implementation

* Python 3.5.2 (later versions than Python 3.5.x are incompatible with TensorFlow’s API)
* TensorFlow API r1.2 
* OS: Windows 7 64-bits or later

## External libraries needed

* Tensorflow
  * https://www.tensorflow.org/install/
* SciPy/Numpy
  * https://scipy.org/install.html 
* PIL
  * http://www.pythonware.com/products/pil/

## Files brief explanation

* style_tranfer_userpromt.py
    * This file contains the main flow from the program. It starts asking the user the specifications for the style transfer, later verifies or downloads a pretrained model, and finally starts to train and optimize the style transfer.

* vgg_model.py
    * Module that contains the main functions for the definition of the vgg convolutional network, calculations of the weights and biases, and the average pool from the layers.

* utils.py
    * Module where is located functions for downloading the pretrained vgg model, resizing the input images, generating the noise in the image, saving the images, calculating the time remaining from the complete procedure, and finally showing the image to the user once the procedure is over.

* data_input.py
    * Module with several functions for validating the input data given by the user.

## Resources
The images (for the content or the styles) included in this project are only for illustrative purposes and are nonprofitable.

### Images for styles

* guernica.jpg
    * Image obtained from https://www.musement.com/es/madrid/abono-paseo-del-arte-con-entradas-al-museo-del-prado-thyssen-bornemisza-y-reina-sofia-2452/
* harlequin.jpg
    * Image obtained from https://en.wikipedia.org/wiki/The_Harlequin%27s_Carnival
* pattern.jpg
    * Image obtained from https://www.shutterstock.com/image-vector/seamless-pattern-colorful-abstracts-translucent-figures-146559467
* starry_night.jpg
    * Image obtained from https://en.wikipedia.org/wiki/The_Starry_Night

### Images for content

* batman.jpg
    * Image obtained from http://noahlc.deviantart.com/art/Batman-Earth-25-Logo-666603110
* f35.jpg
    * Image obtained from https://www.f35.com/media/photos
* deadpool.jpg
    * Image obtained from http://heroichollywood.com/ryan-reynolds-shares-fan-video-combining-deadpool-captain-america-civil-war/

