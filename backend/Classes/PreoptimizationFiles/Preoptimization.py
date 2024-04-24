import os
import sys
import io
import base64

import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder
from werkzeug.utils import secure_filename

from backend.Classes.PreoptimizationFiles import Discretization
from backend.Classes.PreoptimizationFiles import EncodingCategoricalFeatures
from backend.Classes.PreoptimizationFiles import GeneratingPolynomialFeatures
from backend.Classes.PreoptimizationFiles import ImputationOfMissingValues
from backend.Classes.PreoptimizationFiles import NonLinearTransformation
from backend.Classes.PreoptimizationFiles import Normalization
from backend.Classes.PreoptimizationFiles import RemoveData
from backend.Classes.PreoptimizationFiles import Standardization
from backend.Classes.PreoptimizationFiles import Tokenization

from urllib.parse import quote
import logging
import io

class Preopt_Category:
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
list_Preopt_Categories = []

def getPreopt_Categories():
    return list_Preopt_Categories

## Preoptimization options are located at https://scikit-learn.org/stable/modules/preprocessing.html#discretization

# Standardization
Name = "Standardization"
Display_Name = "Standardization"
Definition = ["Standardization of datasets is a common requirement for many machine learning estimators implemented in scikit-learn; they might behave badly if the individual features do not more or less look like standard normally distributed data: Gaussian with zero mean and unit variance.\n\nIn practice we often ignore the shape of the distribution and just transform the data to center it by removing the mean value of each feature, then scale it by dividing non-constant features by their standard deviation.\n\nFor instance, many elements used in the objective function of a learning algorithm (such as the RBF kernel of Support Vector Machines or the l1 and l2 regularizers of linear models) may assume that all features are centered around zero or have variance in the same order. If a feature has a variance that is orders of magnitude larger than others, it might dominate the objective function and make the estimator unable to learn from other features correctly as expected."]

list_Preopt_Categories.append(Preopt_Category(Name, Display_Name, Definition))

# Non-linear Transformation
Name = "NonlinearTransformation"
Display_Name = "Non-linear Transformation"
Definition = ["Two types of transformations are available: quantile transforms and power transforms. Both quantile and power transforms are based on monotonic transformations of the features and thus preserve the rank of the values along each feature.\n\nQuantile transforms put all features into the same desired distribution based on the formula G^(-1)(F(X)) where F is the cumulative distribution function of the feature and G^(-1) the quantile function of the desired output distribution G. This formula is using the two following facts: (i) X if is a random variable with a continuous cumulative distribution function F then is uniformly distributed on F(X); (ii) U if is a random variable with uniform distribution on [0,1] then G^(-1)(U) has distribution G. By performing a rank transformation, a quantile transform smooths out unusual distributions and is less influenced by outliers than scaling methods. It does, however, distort correlations and distances within and across features.\n\nPower transforms are a family of parametric transformations that aim to map data from any distribution to as close to a Gaussian distribution."]

list_Preopt_Categories.append(Preopt_Category(Name, Display_Name, Definition))

# Normalization
Name = "Normalization"
Display_Name = "Normalization"
Definition = ["Normalization is the process of scaling individual samples to have unit norm. This process can be useful if you plan to use a quadratic form such as the dot-product or any other kernel to quantify the similarity of any pair of samples.\n\nThis assumption is the base of the Vector Space Model often used in text classification and clustering contexts."]

list_Preopt_Categories.append(Preopt_Category(Name, Display_Name, Definition))

# Encoding Categorical Features
Name = "EncodingCategoricalFeatures"
Display_Name = "Encoding Categorical Features"
Definition = ['Often features are not given as continuous values but categorical. For example a person could have features ["male", "female"], ["from Europe", "from US", "from Asia"], ["uses Firefox", "uses Chrome", "uses Safari", "uses Internet Explorer"]. Such features can be efficiently coded as integers, for instance ["male", "from US", "uses Internet Explorer"] could be expressed as [0, 1, 3] while ["female", "from Asia", "uses Chrome"] would be [1, 2, 1].']

list_Preopt_Categories.append(Preopt_Category(Name, Display_Name, Definition))

# Discretization
Name = "Discretization"
Display_Name = "Discretization"
Definition = ["Discretization (otherwise known as quantization or binning) provides a way to partition continuous features into discrete values. Certain datasets with continuous features may benefit from discretization, because discretization can transform the dataset of continuous attributes to one with only nominal attributes.\n\nOne-hot encoded discretized features can make a model more expressive, while maintaining interpretability. For instance, pre-processing with a discretizer can introduce nonlinearity to linear models. For more advanced possibilities, in particular smooth ones, see Generating polynomial features further below."]

list_Preopt_Categories.append(Preopt_Category(Name, Display_Name, Definition))

# Imputation of Missing Values
Name = "ImputationofMissingValues"
Display_Name = "Imputation of Missing Values"
Definition = ["For various reasons, many real world datasets contain missing values, often encoded as blanks, NaNs or other placeholders. Such datasets however are incompatible with scikit-learn estimators which assume that all values in an array are numerical, and that all have and hold meaning. A basic strategy to use incomplete datasets is to discard entire rows and/or columns containing missing values. However, this comes at the price of losing data which may be valuable (even though incomplete). A better strategy is to impute the missing values, i.e., to infer them from the known part of the data."]

list_Preopt_Categories.append(Preopt_Category(Name, Display_Name, Definition))

# Generating Polynomial Features
Name = "GeneratingPolynomialFeatures"
Display_Name = "Generating Polynomial Features"
Definition = ["Often it’s useful to add complexity to a model by considering nonlinear features of the input data. We show two possibilities that are both based on polynomials: The first one uses pure polynomials, the second one uses splines, i.e. piecewise polynomials."]

list_Preopt_Categories.append(Preopt_Category(Name, Display_Name, Definition))


# Generating Polynomial Features
Name = "Tokenization"
Display_Name = "Tokenization"
Definition = ["Utility class that automatically selects and loads the appropriate tokenizer based on the provided model."]

list_Preopt_Categories.append(Preopt_Category(Name, Display_Name, Definition))
# This may be implemented later, but is currently to complex for initial implementation of preoptimization
# What is does is allow the user to create a custom function for preoptimization.
# Custom Transformers
#Name = ""
#Display_name = ""
#Definition = [""]

# Remove Data
Name = "RemoveData"
Display_Name = "Remove Data"
Deginition = ["Preoptimization that simply removes information from the dataset."]

list_Preopt_Categories.append(Preopt_Category(Name, Display_Name, Definition))

#list_Preopt_Categories.append(Preopt_Category(Name, Display_Name, Definition))

# Obtain the list of preoptimization from all categories via their separate python files
list_Discretization = Discretization.getDiscretization()
list_ECF = EncodingCategoricalFeatures.getECF()
list_GPF = GeneratingPolynomialFeatures.getGPF()
list_IoMV = ImputationOfMissingValues.getIoMV()
list_NonLT = NonLinearTransformation.getNonLT()
list_Normalization = Normalization.getNormalization()
list_RemoveData = RemoveData.getRemoveData()
list_Standardization = Standardization.getStandardization()
list_tokenization = Tokenization.getTokenization()

# get settings to fill the model parameters.
def getSettings(data, Parameters):
    settings = {}
    settings["Parameter_Length"] = len(Parameters)

    # Cycle through parameters
    for i in range(len(Parameters)):
        temp = Parameters["Parameter_" + str(i)]["Name"]+"_checked"

        # get name
        settings["Parameter_" + str(i) + "_Name"] = Parameters["Parameter_" + str(i)]["Name"]

        # get default
        if data[Parameters["Parameter_" + str(i)]["Name"]+"_checked"] == "false":
            settings["Parameter_" + str(i)] = Parameters["Parameter_" + str(i)]["Default_option"]
            # convert to acceptable type
            settings["Parameter_" + str(i)] = convertToType(settings["Parameter_" + str(i)], Parameters["Parameter_" + str(i)]["Type"])

        elif data[Parameters["Parameter_" + str(i)]["Name"]+"_checked"] == "true":
            temp = Parameters["Parameter_" + str(i)]["Type"]
            if Parameters["Parameter_" + str(i)]["Type"][0] == "option_float":
                settings["Parameter_" + str(i)] = data[Parameters["Parameter_" + str(i)]["Name"] + "_Input"]
                settings["Parameter_" + str(i)] = convertToType(settings["Parameter_" + str(i)], Parameters["Parameter_" + str(i)]["Type"], data[Parameters["Parameter_" + str(i)]["Name"] + "_Float_Input"])
            else:
                settings["Parameter_" + str(i)] = data[Parameters["Parameter_" + str(i)]["Name"] + "_Input"]
                settings["Parameter_" + str(i)] = convertToType(settings["Parameter_" + str(i)], Parameters["Parameter_" + str(i)]["Type"])

    return settings


def convertToType(value, Type, additionalNumber = 0):
    if Type[0] == "int":
        return int(value)
    elif Type[0] == "float":
        return float(value)
    elif Type[0] == "bool":
        if value == "true" or value == True:
            return True
        elif value == "false" or value == False:
            return False
    elif Type[0] == "int_or_null":
        if value == "null" or value == "None" or value == "":
            return None
        elif value != None:
            return int(value)
        else:
            return value
    elif Type[0] == "float_or_null":
        if value == "null" or value == "None" or value == "":
            return None
        elif value != None:
            return float(value)
        else:
            return value
    elif Type[0] == "option":
        if value == "null" or value == "None" or value == "":
            return None
        else:
            return value
    elif Type[0] == "option_float":
        if value == "null" or value == "None" or value == "":
            return None
        else:
            if value == "float":
                return float(additionalNumber)
            else:
                return value
    else:
        return value

# Method to process the proptimization options.


def getDataset(fileName):
     # convert csv to usable dataset
    ## Manual location currently...Will be changed when implemented in Host.
    ## Should further be changed when database is setup.

    baseFolder = os.getcwd()

    ManualLoc = Path(baseFolder) / "backend/tests/sampleCSV_MLA_Classification/"

    filename = secure_filename(fileName)
    dataset_path = ManualLoc / filename.lower()
    dataset = pd.read_csv(dataset_path)

    return dataset

# Return the column titles in an array.
def getCSVColumnTitles(data):
    ##df = getDataset(data['csvFileName'])
    if data['csvFileName'].endswith('.npy'):
        numpy_array = np.load(data['csvFile'])
        # Convert the NumPy array to a DataFrame
        df = pd.DataFrame(data=numpy_array[1:, :], columns=numpy_array[0, :])
    elif data['csvFileName'].endswith('.csv'):
        # logging.debug("Hello")
        df = pd.read_csv(data['csvFile'], index_col=None)

    titles_list = list(df.columns.values)
    # titles_list.append("")
    results = {"Tiltes":titles_list}

    return results

# Returns all preoptimizations from a Category
def getCategoryPreopts(Category_Name):
    info = []
    if Category_Name == "Discretization":
        preopts = list_Discretization
    if Category_Name == "EncodingCategoricalFeatures":
        preopts = list_ECF
    if Category_Name == "GeneratingPolynomialFeatures":
        preopts = list_GPF
    if Category_Name == "ImputationofMissingValues":
        preopts = list_IoMV
    if Category_Name == "NonlinearTransformation":
        preopts = list_NonLT
    if Category_Name == "Normalization":
        preopts = list_Normalization
    if Category_Name == "RemoveData":
        preopts = list_RemoveData
    if Category_Name == "Standardization":
        preopts = list_Standardization
    if Category_Name == "Tokenization":
        preopts = list_tokenization

    for preopt in preopts:
        preopt_info = {"Name": preopt.getName(), "Display_Name": preopt.getDisplayName()}
        info.append(preopt_info)

    return info

def getCSV_PDF(data):
    try: 
        # convert csv to usable dataset
        ##df = getDataset(data['csvFileName'])
        if data['csvFileName'].endswith('.npy'):
            numpy_array = np.load(data['csvFile'])
            # Convert the NumPy array to a DataFrame
            df = pd.DataFrame(data=numpy_array[1:, :], columns=numpy_array[0, :])
        elif data['csvFileName'].endswith('.csv'):
        # logging.debug("Hello")
            df = pd.read_csv(data['csvFile'], index_col=None)

        # perform preoptimization
        if data["Perform_Preopt"] == "Yes":
            # Cycle through the choices
            for i in range(int(data["preoptCounter"])):
                # Get the choice of preoptimization.
                preopt_method = data["Preopt_" + str(i)]
                # Perform the preoptimization on the dataset.
                df = perform_Preopt(data, i, df)

        dataframe = df.to_html(max_rows=10, show_dimensions=True)

        null_count = df.isnull().sum().rename_axis('Column').reset_index(name='Number of Null Instances')
        null_count = null_count.to_html()

        index_of_class_col = df.columns.get_loc(data["class_col"])

        number_classes = ""

        # stream = io.StringIO()
        # df.dtypes.to_string(buf=stream)
        # df_dtypes = stream.getvalue().replace('\n', '<br>')
        # data_type = df_dtypes
        
        df_dtypes = df.dtypes.to_frame().rename(columns={0: 'Data Type'})
        data_type = df_dtypes.to_html()

        if data["class_col"] != None and data["class_col"] != "": 
            number_classes = df[df.columns[index_of_class_col]].value_counts().rename_axis('Unique Values').reset_index(name='counts')
            number_classes = number_classes.to_html()
            shape = df.shape
    
        kde_base64 = []
        if data["kde_ind"] != None and data["kde_ind"] != "": 
            if data["class_col"] != None and data["class_col"] != "": 
                classes = df[df.columns[index_of_class_col]].unique()

                X = df.loc[:, df.columns != df.columns[index_of_class_col]].values.astype('float64')
                Y = df[df.columns[index_of_class_col]].values

                for class_name in classes:
                    temp = X[Y == class_name]
                    pd.DataFrame(X[Y == class_name]).plot.kde(ind=int(data["kde_ind"]), legend=True)
                    plt.legend(loc='best', title='Column')
                    title = 'Likelihood KDE Plots for the ' + str(class_name) + ' Class'
                    plt.title(title);

                    plt.plot()
                    my_stringIObytes = io.BytesIO()
                    plt.savefig(my_stringIObytes, format='jpg')
                    my_stringIObytes.seek(0)
                    my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode()

                    kde_base64.append(my_base64_jpgData)

        PDF = { "csv_Short": dataframe,
                "shape": shape,
                "null_Count": null_count,
                "Number_Classes": number_classes,
                "kde_plots": kde_base64,
                "data_type": data_type}
    
        status = "worked"
        msg = ""

        return status, msg, PDF

    except Exception as e:
        PDF = ""
        msg = str(e)
        status = "error"

        return status, msg, PDF

def downloadCSV(data):
    try: 
        # convert csv to usable dataset
        ##df = getDataset(data['csvFileName'])
        df = pd.read_csv(data['csvFile'], index_col=None)

        # perform preoptimization
        if data["Perform_Preopt"] == "Yes":
            # Cycle through the choices
            for i in range(int(data["preoptCounter"])):
                # Get the choice of preoptimization.
                preopt_method = data["Preopt_" + str(i)]
                # Perform the preoptimization on the dataset.
                df = perform_Preopt(data, i, df)

        csv_data = build_csv_data(df)

        PDF = { "csv_data": csv_data }
    
        status = "worked"
        msg = ""

        return status, msg, PDF

    except Exception as e:
        PDF = ""
        msg = str(e)
        status = "error"

        return status, msg, PDF

def build_csv_data(dataframe):
    csv_data = dataframe.to_csv(index=False, encoding='utf-8')
    csv_data = "data:text/csv;charset=utf-8," + quote(csv_data)
    return csv_data

# Method to pull all parameters from a given preoptimization option
def getParameters(Preopt_Name):
    Parameters = {}
    #list_Discretization = Discretization.getDiscretization()
    #list_ECF = EncodingCategoricalFeatures.getECF()
    #list_GPF = GeneratingPolynomialFeatures.getGPF()
    #list_IoMV = ImputationOfMissingValues.getIoMV()
    #list_NonLV = NonLinearTransformation.getNonLT()
    #list_Normalization = Normalization.getNormalization()
    #list_RemoveData = RemoveData.getRemoveData()
    #list_Standardization = Standardization.getStandardization()
    
    for preopt in list_Discretization:
        if preopt.getName() == Preopt_Name:
            Parameters = preopt.getParameters()
            return Parameters

    for preopt in list_ECF:
        if preopt.getName() == Preopt_Name:
            Parameters = preopt.getParameters()
            return Parameters

    for preopt in list_GPF:
        if preopt.getName() == Preopt_Name:
            Parameters = preopt.getParameters()
            return Parameters

    for preopt in list_IoMV:
        if preopt.getName() == Preopt_Name:
            Parameters = preopt.getParameters()
            return Parameters

    for preopt in list_NonLT:
        if preopt.getName() == Preopt_Name:
            Parameters = preopt.getParameters()
            return Parameters

    for preopt in list_Normalization:
        if preopt.getName() == Preopt_Name:
            Parameters = preopt.getParameters()
            return Parameters

    for preopt in list_RemoveData:
        if preopt.getName() == Preopt_Name:
            Parameters = preopt.getParameters()
            return Parameters

    for preopt in list_Standardization:
        if preopt.getName() == Preopt_Name:
            Parameters = preopt.getParameters()
            return Parameters
        
    for preopt in list_tokenization:
        if preopt.getName() == Preopt_Name:
            Parameters = preopt.getParameters()
            return Parameters


    return Parameters

# Method to check the chosen preoptimization method and perform it on the dataset.
def perform_Preopt(data, i, df):
    # get method
    method = data["Preopt_" + str(i)]

    new_df = df

    found = 0

    ## Standardization options
    if found == 0:
        for preopts in list_Standardization:
            if method == preopts.getName():
                new_df = Standardization.perform_Preopt(data, i, df)
                found = 1
                break

    ## NonLinear Transformation options
    if found == 0:
        for preopts in list_NonLT:
            if method == preopts.getName():
                new_df = NonLinearTransformation.perform_Preopt(data, i, df)
                found = 1
                break

    ## Normalization
    if found == 0:
        for preopts in list_Normalization:
            if method == preopts.getName():
                new_df = Normalization.perform_Preopt(data, i, df)
                found = 1
                break

    ## Encoding Categorical Features
    if found == 0:
        for preopts in list_ECF:
            if method == preopts.getName():
                new_df = EncodingCategoricalFeatures.perform_Preopt(data, i, df)
                found = 1
                break

    ## Discretization
    if found == 0:
        for preopts in list_Discretization:
            if method == preopts.getName():
                new_df = Discretization.perform_Preopt(data, i, df)
                found = 1
                break

    ## Imputation of Missing Values
    if found == 0:
        for preopts in list_IoMV:
            if method == preopts.getName():
                new_df = ImputationOfMissingValues.perform_Preopt(data, i, df)
                found = 1
                break

    ## Generating Polynomial Features
    if found == 0:
        for preopts in list_GPF:
            if method == preopts.getName():
                new_df = GeneratingPolynomialFeatures.perform_Preopt(data, i, df)
                found = 1
                break

    ## RemoveData
    if found == 0:
        for preopts in list_RemoveData:
            if method == preopts.getName():
                new_df = RemoveData.perform_Preopt(data, i, df)
                found = 1
                break

    return new_df