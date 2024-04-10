import base64

import pandas as pd
import numpy as np

from tensorflow import keras
from backend.Classes.NeuralNetworkFiles import Callbacks

##
Name = "Validation"
Display_Name = "Validation"
Definition = ["Options for the model to fit, validate and train the model."]
Parameter_0 = {"Name":"test_split", "Type": ["float"], "Default_option":"0.1", "Default_value":"0.1", "Possible":["float"],
             "Definition":"Float number given splits the dateset by the given amount to create the test set. I.E. 0.1 will create a test set containing 10% of the given dataset."}
Parameter_1 = {"Name":"validation_split", "Type": ["float"], "Default_option":0.0, "Default_value":0.0, "Possible":["float"],
             "Definition":"Float number given splits the remaining training set by the given amount to create a validation set. I.E. 0.1 will create a validation set containing 10% of the given training set."}
Parameter_2 = {"Name":"random_state", "Type": ["int"], "Default_option":"", "Default_value":"", "Possible":["int"],
             "Definition":"The random seed that will be used when utalizing any random number caller. Helps in obtaining repeatable results."}
Parameter_3 = {"Name":"shuffle_before_split", "Type": ["bool"], "Default_option":False, "Default_value":False, "Possible":[True, False],
             "Definition":"When set to True, the dataset will be shuffled before any splitting for creating the test or validation sets occures."}
Parameter_4 = {"Name":"shuffle_before_epoch", "Type": ["bool"], "Default_option":False, "Default_value":False, "Possible":[True, False],
             "Definition":"When set to True, the dataset to train the model will be shuffled before each epoch."}
Parameter_5 = {"Name":"epochs", "Type": ["int"], "Default_option":10, "Default_value":10, "Possible":["int"],
             "Definition":'Integer. Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided (unless the steps_per_epoch flag is set to something other than None). Note that in conjunction with initial_epoch, epochs is to be understood as "final epoch". The model is not trained for a number of iterations given by epochs, but merely until the epoch of index epochs is reached.'}
Parameter_6 = {"Name":"batch_size", "Type": ["int"], "Default_option":"", "Default_value":"", "Possible":["int"],
             "Definition":"Integer or None. Number of samples per gradient update. If unspecified, batch_size will default to 32. Do not specify the batch_size if your data is in the form of datasets, generators, or keras.utils.Sequence instances (since they generate batches)."}
Parameter_7 = {"Name":"verbose", "Type": ["option"], "Default_option":"auto", "Default_value":"auto", "Possible":["auto","0","1","2"],
             "Definition":"'auto', 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. 'auto' becomes 1 for most cases, but 2 when used with ParameterServerStrategy. Note that the progress bar is not particularly useful when logged to a file, so verbose=2 is recommended when not running interactively (eg, in a production environment). Defaults to 'auto'."}

Parameters = {"Parameter_0":Parameter_0, "Parameter_1":Parameter_1, "Parameter_2":Parameter_2, "Parameter_3":Parameter_3,
              "Parameter_4":Parameter_4, "Parameter_5":Parameter_5, "Parameter_6":Parameter_6,
              "Parameter_7":Parameter_7}

Validation = {"Name": Name, "Display_Name": Display_Name, "Definition": Definition, "Parameters": Parameters}

def getValidationOptions():
    return Validation

def fitModel(model, data, x_train, y_train):
    if data["Validation_batch_size_Input"] == "":
        batch_size = None
    else:
        batch_size = int(data["Validation_batch_size_Input"])
    epochs = int(data["Validation_epochs_Input"])

    if data["Validation_verbose_Input"] == "auto":
        verbose = "auto"
    else:
        verbose = int(data["Validation_verbose_Input"])

    callbacks= Callbacks.createCallbacks(data)

    validation_split= float(data["Validation_validation_split_Input"])

    if data["Validation_shuffle_before_epoch_Input"] == "true":
        shuffle = True
    else:
        shuffle = False



    history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks,
              validation_split=validation_split, shuffle=shuffle)

    return history