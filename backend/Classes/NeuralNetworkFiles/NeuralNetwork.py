import os
import sys
import io
import base64

import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

from backend.Classes.NeuralNetworkFiles import Core
from backend.Classes.NeuralNetworkFiles import Convolution
from backend.Classes.NeuralNetworkFiles import Normalization
from backend.Classes.NeuralNetworkFiles import Regularization
from backend.Classes.NeuralNetworkFiles import Activation
from tensorflow import keras

from io import StringIO

class Layer_Category:
    def __init__(self, name, display_name, definition):
        self.name = name
        self.display_name = display_name
        self.definition = definition
    
    def getName(self):
        return self.name

    def getDisplayName(self):
        return self.display_name

    def getDefinition(self):
        return self.definition

# Create a list of possible categories to sect from.
list_Layer_Categories = []

def getLayer_Categories():
    return list_Layer_Categories

# Layer options are located at https://keras.io/api/layers/

## Core
Name = "Core"
Display_Name = "Core"
Definition = ["Core layers that most Neural Networks require."]

list_Layer_Categories.append(Layer_Category(Name, Display_Name, Definition))

## Convolution layers
Name = "Convolution"
Display_Name = "Convolution"
Definition = ["Convolution layers."]

list_Layer_Categories.append(Layer_Category(Name, Display_Name, Definition))

## Normalization layers
Name = "Normalization"
Display_Name = "Normalization"
Definition = ["Normalization layers."]

list_Layer_Categories.append(Layer_Category(Name, Display_Name, Definition))

## Regularization layers
Name = "Regularization"
Display_Name = "Regularization"
Definition = ["Regularization layers."]

list_Layer_Categories.append(Layer_Category(Name, Display_Name, Definition))


Name = "Activation"
Display_Name = "Activation"
Definition = ["Activation layers."]

list_Layer_Categories.append(Layer_Category(Name, Display_Name, Definition))


# obtain the list of Layers from all categories via their separate python files
list_Core = Core.getCore()
list_Convolution = Convolution.getConvolution()
list_Normalization = Normalization.getNormalization()
list_Regularization = Regularization.getRegularization()
list_Activation = Activation.getActivation()


# Methods to process the Neural Network options.

# Returns all layers from a Category
def getCategoryLayers(Category_Name):
    info = []
    if Category_Name == "Core":
        layers = list_Core

    if Category_Name == "Convolution":
        layers = list_Convolution

    if Category_Name == "Normalization":
        layers = list_Normalization
    
    if Category_Name == "Regularization":
        layers = list_Regularization
    
    if Category_Name == "Activation":
        layers = list_Activation

    for layer in layers:
        layer_info = {"Name": layer.getName(), "Display_Name": layer.getDisplayName()}
        info.append(layer_info)

    return info

# Method to pull all parameters from a given layer option
def getParameters(Layer_Name):
    Parameters = {}

    # Check Core
    for layer in list_Core:
        if layer.getName() == Layer_Name:
            Parameters = layer.getParameters()
            return Parameters

    # Check Convolution
    for layer in list_Convolution:
        if layer.getName() == Layer_Name:
            Parameters = layer.getParameters()
            return Parameters

    # Check Normalization
    for layer in list_Normalization:
        if layer.getName() == Layer_Name:
            Parameters = layer.getParameters()
            return Parameters
        
    # Check Regularization
    for layer in list_Regularization:
        if layer.getName() == Layer_Name:
            Parameters = layer.getParameters()
            return Parameters
    
    for layer in list_Activation:
        if layer.getName() == Layer_Name:
            Parameters = layer.getParameters()
            return Parameters
        
    return Parameters

# Method to create a neural network layer
def get_Layer(data, i):
    new_layer = ""

    # get layer
    layer = data["Layer_" + str(i)]

    found = 0

    ## Core options
    if found == 0:
        for layers in list_Core:
            if layer == layers.getName():
                new_layer = Core.create_Layer(data, i)
                found = 1
                break

    ## Convolution options
    if found == 0:
        for layers in list_Convolution:
            if layer == layers.getName():
                new_layer = Convolution.create_Layer(data, i)
                found = 1
                break

    ## Normalization options
    if found == 0:
        for layers in list_Normalization:
            if layer == layers.getName():
                new_layer = Normalization.create_Layer(data, i)
                found = 1
                break

    ## Regularization options
    if found == 0:
        for layers in list_Regularization:
            if layer == layers.getName():
                new_layer = Regularization.create_Layer(data, i)
                found = 1
                break
    


    if found == 0:
        for layers in list_Activation:
            if layer == layers.getName():
                new_layer = Activation.create_Layer(data, i)
                found = 1
                break
    

    return new_layer

# Method to create the model based on data information given.
def createModel(data):
    model = ""
    settings = []
    
    layers = []

    # Cycle through the choices
    for i in range(int(data["layerCounter"])):
        # Get the choice of neural network layer.
        layer_Option = data["Layer_" + str(i)]
        # get the layer and place in the list.
        layers.append(get_Layer(data, i))

    # Cycle through list of layers and create model.
    if len(layers) > 0:
        model = keras.Sequential()
        for i in range(len(layers)):
            model.add(layers[i])

    return model

# Method to return model summary
def getModelSummary(data):
    try: 
        # convert information provided in data into the model.
        model = createModel(data)

        tmp_smry = StringIO()
        model.summary(print_fn=lambda x: tmp_smry.write(x + '\n'))
        model_summary = tmp_smry.getvalue()
        model_summary = model_summary.replace('\n', '<br>')

        return_data = { "model_summary": model_summary }
        status = "worked"
        msg = ""

        return status, msg, return_data

    except Exception as e:
        return_data = ""
        msg = str(e)
        status = "error"

        return status, msg, return_data

# Method to check the chosen neural network and return a summary.
def perform_Preopt(data, i):
    result = ""
    return result