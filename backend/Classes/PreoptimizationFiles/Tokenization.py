import os
import sys
import io
import base64

import pandas as pd
import numpy as np

from backend.Classes.PreoptimizationFiles import Preoptimizer
from backend.Classes import HyperParameters

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.feature_selection import VarianceThreshold

# Create a list of standardization preoptimizations to select from.
list_TokenizeData = []

def getTokenization():
    return list_TokenizeData

## Drop Columns:
Name = "BERT"
Display_Name = "BERT"
Definition = ["Tokenization for BERT"]
Parameter_0 = {"Name":"max_length", "Type": ["int"], "Default_option":"", "Default_value":"", "Possible":["int"],
             "Definition":"Sequences longer than this length will be truncated (if truncation=True) or discarded."}

Parameter_1 = {"Name":"Truncation", "Type": ["bool"], "Default_option":True, "Default_value":True, "Possible":[True, False], 
             "Definition":"If set to True, truncates the sequences to a maximum length specified by the max_length parameter."}

Parameters = {"Parameter_0":Parameter_0, "Parameter_1": Parameter_1 }

list_TokenizeData.append(Preoptimizer.Preoptimizer(Name, Display_Name, Definition, Parameters))

# ## Drop Rows:
# Name = "Truncation"
# Display_Name = "Truncation"
# Definition = ["Remove row from the dataset."]
# Parameter_0 = {"Name":"row", "Type": ["int"], "Default_option":"", "Default_value":"", "Possible":["int"],
#              "Definition":"If set to True, truncates the sequences to a maximum length specified by the max_length parameter."}

# Parameters = {"Parameter_0":Parameter_0}

# list_RemoveData.append(Preoptimizer.Preoptimizer(Name, Display_Name, Definition, Parameters))

# ## Drop null values Removal:
# Name = "DropRowsContainingNullValues"
# Display_Name = "Drop Rows Containing Null Values"
# Definition = ["Removes the rows that contain null value entires in any of the columns."]

# Parameters = {}

# list_RemoveData.append(Preoptimizer.Preoptimizer(Name, Display_Name, Definition, Parameters))

# # VarianceThreshold
# Name = "VarianceThreshold"
# Display_Name = "Variance Threshold"
# Definition = ["Feature selector that removes all low-variance features.\n\nThis feature selection algorithm looks only at the features (X), not the desired outputs (y), and can thus be used for unsupervised learning."]

# Parameter_0 = {"Name":"threshold", "Type": ["float"], "Default_option":0.0, "Default_value":"", "Possible":["float"],
#              "Definition":"Features with a training-set variance lower than this threshold will be removed. The default is to keep all features with non-zero variance, i.e. remove the features that have the same value in all samples."}

# Parameters = {"Parameter_0":Parameter_0}

# list_RemoveData.append(Preoptimizer.Preoptimizer(Name, Display_Name, Definition, Parameters))

def perform_Preopt(data, i, df):
    # get method
    method = data["Preopt_" + str(i)]

    new_df = df.copy()

    Parameters = HyperParameters.getParameters(data["Preopt_" + str(i)], list_RemoveData)

    ## Removal
    if method == "RemoveColumn":
        new_df = df.drop(columns=[data["Preopt_" + str(i) + "_column_Input"]])

    if method == "RemoveRow":
        new_df = df.drop([int(data["Preopt_" + str(i) + "_row_Input"])])

    ## Null instances
    ### Remove all
    if method == "DropRowsContainingNullValues":
        new_df = df.dropna(axis=0)

    if method == "VarianceThreshold":
        threshold = float(data["Preopt_" + str(i) + "_threshold_Input"])

        pp = VarianceThreshold(threshold=threshold)

        new_df = pd.DataFrame(pp.fit_transform(new_df.loc[:, new_df.columns != data["class_col"]]))

        new_df = new_df.join(df[data["class_col"]])
        
        columnTitles = []

        for i in range(new_df.shape[1]-1):
            for name in list(df.columns.values):
                if (new_df[i] == df[name]).all():
                    columnTitles.append(name)
                    break

        columnTitles.append(data["class_col"])

        new_df.columns = columnTitles

    return new_df
