# Final Project
Handwritten Character Recognition Based on Multi-Layer Perception
The source data is 300X300 npy image file as well as feature labels.
Due to limit data storage of github, i don't push any training data and test data. U can just refer the code and replace it with your own data.
If you need to run the training file, you can run the preprocess file first and then run training file. 



## Required Pkg
```
import numpy as np
import numpy.random as npr
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('bmh')
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 28}

matplotlib.rc('font', **font)

import cv2
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import time

from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import regularizers
from tensorflow.keras import layers
```
