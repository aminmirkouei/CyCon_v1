import base64

import pandas as pd
import numpy as np

from tensorflow import keras

loss = ["binary_crossentropy","categorical_crossentropy","sparse_categorical_crossentropy","poisson","kl_divergence","mean_squared_error",
        "mean_squared_error","mean_absolute_error","mean_absolute_percentage_error","mean_squared_logarithmic_error","cosine_similarity", 
        "huber","log_cosh","None"]

## Input
Name = "Compiler"
Display_Name = "Compiler"
Definition = ["Options to compile the determined model."]
Parameter_0 = {"Name":"optimizer", "Type": ["option"], "Default_option":"rmsprop", "Default_value":"rmsprop", "Possible":["adadelta", "adam", "adamax", "ftrl", "nadam", "sgd", "rmsprop"],
             "Definition":"String (name of optimizer) or optimizer instance."}
# This may be added later to show graph of history. For now, this is not implemented as there may be better way to do with with the choices of the model Callbacks.
#Parameter_1 = {"Name":"metrics", "Type": ["option_select"], "Default_option":"accuracy", "Default_value":"accuracy", "Possible":["int"],
#             "Definition":"List of metrics to be evaluated by the model during training and testing. Each of this can be a string (name of a built-in function), function or a tf.keras.metrics.Metric instance. See tf.keras.metrics. Typically you will use metrics=['accuracy']. A function is any callable with the signature result = fn(y_true, y_pred). To specify different metrics for different outputs of a multi-output model, you could also pass a dictionary, such as metrics={'output_a':'accuracy', 'output_b':['accuracy', 'mse']}. You can also pass a list to specify a metric or a list of metrics for each output, such as metrics=[['accuracy'], ['accuracy', 'mse']] or metrics=['accuracy', ['accuracy', 'mse']]. When you pass the strings 'accuracy' or 'acc', we convert this to one of tf.keras.metrics.BinaryAccuracy, tf.keras.metrics.CategoricalAccuracy, tf.keras.metrics.SparseCategoricalAccuracy based on the shapes of the targets and of the model output. We do a similar conversion for the strings 'crossentropy' and 'ce' as well. The metrics passed here are evaluated without sample weighting; if you would like sample weighting to apply, you can specify your metrics via the weighted_metrics argument instead."}
Parameter_1 = {"Name":"loss", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":loss,
             "Definition":"Loss function. May be a string (name of loss function), or a tf.keras.losses.Loss instance. See tf.keras.losses. A loss function is any callable with the signature loss = fn(y_true, y_pred), where y_true are the ground truth values, and y_pred are the model's predictions. y_true should have shape (batch_size, d0, .. dN) (except in the case of sparse loss functions such as sparse categorical crossentropy which expects integer arrays of shape (batch_size, d0, .. dN-1)). y_pred should have shape (batch_size, d0, .. dN). The loss function should return a float tensor. If a custom Loss instance is used and reduction is set to None, return value has shape (batch_size, d0, .. dN-1) i.e. per-sample or per-timestep loss values; otherwise, it is a scalar. If the model has multiple outputs, you can use a different loss on each output by passing a dictionary or a list of losses. The loss value that will be minimized by the model will then be the sum of all individual losses, unless loss_weights is specified."}
# Removed do to complexity of having user deside loss_weights given a matrix of the outputs
#Parameter_2 = {"Name":"loss_weights", "Type": ["option"], "Default_option":"", "Default_value":"", "Possible":["int"],
#             "Definition":"Optional list or dictionary specifying scalar coefficients (Python floats) to weight the loss contributions of different model outputs. The loss value that will be minimized by the model will then be the weighted sum of all individual losses, weighted by the loss_weights coefficients. If a list, it is expected to have a 1:1 mapping to the model's outputs. If a dict, it is expected to map output names (strings) to scalar coefficients."}
#Parameter_3 = {"Name":"weighted_metrics", "Type": ["int"], "Default_option":"", "Default_value":"", "Possible":["int"],
#             "Definition":"List of metrics to be evaluated and weighted by sample_weight or class_weight during training and testing."}
Parameter_2 = {"Name":"run_eagerly", "Type": ["bool"], "Default_option":"None", "Default_value":"None", "Possible":[True,False,"None"],
             "Definition":"Bool. If True, this Model's logic will not be wrapped in a tf.function. Recommended to leave this as None unless your Model cannot be run inside a tf.function. run_eagerly=True is not supported when using tf.distribute.experimental.ParameterServerStrategy. Defaults to False."}
Parameter_3 = {"Name":"steps_per_execution", "Type": ["int"], "Default_option":"None", "Default_value":"None", "Possible":["int"],
             "Definition":"An optional name string for the layer. Should be unique in a model (do not reuse the same name twice). It will be autogenerated if it isn't provided."}
Parameter_4 = {"Name":"jit_compile", "Type": ["bool"], "Default_option":"None", "Default_value":"None", "Possible":[True,False,"None"],
             "Definition":"If True, compile the model training step with XLA. XLA is an optimizing compiler for machine learning. jit_compile is not enabled for by default. Note that jit_compile=True may not necessarily work for all models. For more information on supported operations please refer to the XLA documentation."}
# For some reason this is not a parameters of compile...even though it is stated as such on the keras web page.
#Parameter_5 = {"Name":"pss_evaluation_shards", "Type": ["option_int"], "Default_option":0, "Default_value":0, "Possible":["auto","int"],
#             "Definition":"Integer or 'auto'. Used for tf.distribute.ParameterServerStrategy training only. This arg sets the number of shards to split the dataset into, to enable an exact visitation guarantee for evaluation, meaning the model will be applied to each dataset element exactly once, even if workers fail. The dataset must be sharded to ensure separate workers do not process the same data. The number of shards should be at least the number of workers for good performance. A value of 'auto' turns on exact evaluation and uses a heuristic for the number of shards based on the number of workers. 0, meaning no visitation guarantee is provided. NOTE: Custom implementations of Model.test_step will be ignored when doing exact evaluation. Defaults to 0"}

Parameters = {"Parameter_0":Parameter_0, "Parameter_1":Parameter_1, "Parameter_2":Parameter_2, "Parameter_3":Parameter_3,
              "Parameter_4":Parameter_4}

Compile = {"Name": Name, "Display_Name": Display_Name, "Definition": Definition, "Parameters": Parameters}

def getCompilerOptions():
    return Compile

def compileModel(model, data):
    if data["Compiler_run_eagerly_Input"] == "None":
        run_eagerly = None
    elif data["Compiler_run_eagerly_Input"] == "true":
        run_eagerly = True
    else:
        run_eagerly = False

    if data["Compiler_jit_compile_Input"] == "None":
        jit_compile = None
    elif data["Compiler_jit_compile_Input"] == "true":
        jit_compile = True
    else:
        jit_compile = False

    model.compile(optimizer=data["Compiler_optimizer_Input"], loss=data["Compiler_loss_Input"], metrics=['accuracy'],
                  run_eagerly=run_eagerly, steps_per_execution=data["Compiler_steps_per_execution_Input"], jit_compile=jit_compile) 
