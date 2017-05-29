# MNIST Classifier
This repository contains MNIST classifier using

1. MLP
2. CNN

Both achieve an accuracy of >98% (CNN is slightly better). Implemented in Keras with a Tensorflow backend. There are two sets of pre-trained weights for each.

The files perform the following function
1. mnist_classifier.py - conatins models and training algorithms
2. mnist_predictor.py - uses stored weights , picks 10 images at random and creates jpegs with "value(prediction)" on top of numbers , the jpegs are essentially visualized predictions

pre-trained weights with **orig** tag in their file name , were trained on nvidia gtx 1080 GPU based on the models.

Required packages can be found in requirements.txt
