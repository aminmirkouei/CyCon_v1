import base64

import pandas as pd
import numpy as np

from backend.Classes.NeuralNetworkFiles import Layer
from backend.Classes import HyperParameters

import matplotlib.pyplot as plt

from tensorflow import keras

# Create a list of core layers to select from.
list_Activation = []

def getActivation():
    return list_Activation

## ReLU Layer
Name = "ReLU"
Display_Name = "ReLU"
Definition = ['Rectified Linear Unit activation function.'],

Parameter_0 = {"Name":"max_value", 
               "Type": ["float"], 
               "Default_option":"", 
               "Default_value":"", 
               "Possible":["float"],
             "Definition":"Maximum activation value. None means unlimited. Defaults to None."}

# Only allows one number to reshape into a 1d array of (number, )
# may have to allow for more dimentions later.
Parameter_1 = {"Name":"negative_slope", 
               "Type": ["float"], 
               "Default_option":"0.0", 
               "Default_value":"0.0", 
               "Possible":["float"],
             "Definition":"Negative slope coefficient. Defaults to 0."}

Parameter_2 = {"Name":"threshold", 
               "Type": ["float"], 
               "Default_option":"0.0", 
               "Default_value":"0.0", 
               "Possible":["float"],
             "Definition":"Threshold value for thresholded activation. Defaults to 0."}

Parameters = {"Parameter_0":Parameter_0, "Parameter_1":Parameter_1, "Parameter_2":Parameter_2}

list_Activation.append(Layer.Layer(Name, Display_Name, Definition, Parameters))




## Softmax Layer
Name = "Softmax"
Display_Name = "Softmax"
Definition = ['Softmax activation function.'],

Parameter_0 = {"Name":"axis", 
               "Type": ["integer"], 
               "Default_option":"-1", 
               "Default_value":"-1", 
               "Possible":["integer"],
             "Definition":"Axis along which the softmax normalization is applied."}


Parameters = {"Parameter_0":Parameter_0}

list_Activation.append(Layer.Layer(Name, Display_Name, Definition, Parameters))


Name = "LeakyReLU"
Display_Name = "LeakyReLU"
Definition = ['Leaky version of a Rectified Linear Unit.'],

Parameter_0 = {"Name":"alpha", 
               "Type": ["float"], 
               "Default_option":"0.3", 
               "Default_value":"0.3", 
               "Possible":["float"],
             "Definition":"Float >= 0.. Negative slope coefficient. Defaults to 0.3."}


Parameters = {"Parameter_0":Parameter_0}

list_Activation.append(Layer.Layer(Name, Display_Name, Definition, Parameters))


Name = "ELU"
Display_Name = "ELU"
Definition = ['Exponential Linear Unit.'],

Parameter_0 = {"Name":"alpha", 
               "Type": ["float"], 
               "Default_option":"1.0", 
               "Default_value":"1.0", 
               "Possible":["float"],
             "Definition":"Scale for the negative factor."}


Parameters = {"Parameter_0":Parameter_0}

list_Activation.append(Layer.Layer(Name, Display_Name, Definition, Parameters))




def create_Layer(data, i):
    # get layer name
    layer = data["Layer_" + str(i)]

    # get the chosen settings for the layer
    Parameters = HyperParameters.getParameters(data["Layer_" + str(i)], list_Activation)
    settings = HyperParameters.getSettings(data, Parameters, i, Layer.getName())

    new_layer = ""

    ## Dropout Layer
    if layer == "ReLU":
        # Create the layer
        new_layer = keras.layers.ReLU(max_value=settings["Parameter_0"], negative_slope=settings["Parameter_1"], threshold=settings["Parameter_2"])
    
    if layer == "LeakyReLU":
        # Create the layer
        new_layer = keras.layers.LeakyReLU(alpha=settings["Parameter_0"])

    if layer == "ELU":
        # Create the layer
        new_layer = keras.layers.ELU(alpha=settings["Parameter_0"])
    
    
        
    return new_layer