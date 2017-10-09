from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_yaml
from keras.models import load_model
import numpy
import os
#load YAML and create the model
#yaml_file = open('mo')

# load model
model = load_model('nn.h5')

lstmweights = model.get_weights()
print(lstmweights)
