import os
import sys
import csv
from pathlib import Path
import json
from flask import Blueprint
from flask import Flask, request, render_template, g, redirect, url_for, flash
from flask import session
from mysql.connector import Error
from flask import current_app
from flask import jsonify
import logging
from pathlib import Path

from sympy.printing.latex import latex
import sympy

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, f1_score, precision_score

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

# Models
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from werkzeug.utils import secure_filename

from backend.calculator import Calculator, read_data

from backend.db import DBManager

# sys.path.append("backend")
# sys.path.append("Classes")
# sys.path.insert(1, '/Classes/')
from backend import MLA
from backend import Ensemble
from backend import MLA_Validation
from backend import DLANN_Validation
from backend import Ensemble_Validation
from backend.Classes.PreoptimizationFiles import Preoptimization
from backend.Classes.PreoptimizationFiles import Standardization
from backend.Classes.NeuralNetworkFiles import NeuralNetwork
from backend.Classes.NeuralNetworkFiles import Core
from backend.Classes.NeuralNetworkFiles import Compiler
from backend.Classes.NeuralNetworkFiles import Validation as NN_Validation
from backend.Classes.NeuralNetworkFiles import Callbacks

from keras.utils import to_categorical 

from sklearn.model_selection import KFold

bp = Blueprint("experiments", __name__, url_prefix="/experiments")


logger = logging.getLogger()

# should be modified to work with a db or class if necessary.
# Otherwise, the choices being pre-available on page load will slightly speed the process.
#  etting up some initial state before the actual view function is executed.
@bp.before_app_request
def load_possible_experiments():
    list_Algorithms = MLA.getMLAs()
    list_Ensembles = Ensemble.getEnsembles()
    # list for preoptimization options
    list_Preopt_Categories = Preoptimization.getPreopt_Categories()  # Just display name, display_name, definition of all
    list_Standardization = Standardization.getStandardization()      # Name,definition and parameters. for all standardization type
    # list for Neural Network options
    list_Layer_Categories = NeuralNetwork.getLayer_Categories()  #
    list_Core = Core.getCore()
    # list for Callback options
    list_Callbacks = Callbacks.getCallbacks()

    #----------------Data Types-------------------------->
    g.data_types = ["Tabular", "Image", "Text", "Signal"]
    g.data_types_Name = ["Tabular", "Image", "Text", "Signal"]
    g.data_types_Info = [["Tabular is data that is displayed in columns or tables and does not pertain to the other data types."],
                      ["Image is data that is usually 3-dimentional corresponding to the three color spectrom, RGB"],
                      ["Text is data containing textual words or sentences."],
                      ["Signal is data containing time-series data corresponding to a signal format."]]
    g.Data_Types = zip(g.data_types, g.data_types_Name, g.data_types_Info)

    #----------------new Cycon--------------------------->
    # First choice is between various categories of ML 
    g.section_Method = ["MLA", "DLANN", "NLP"]
    g.section_Method_Display_Name = ["Machine Learning Algorithm (MLA)", "Deep Learning Artifical Neural Networks (DLANN)", "Natural Language Processing"]
    g.section_Info = [["MLAs are mathematical tools that enable machines to learn from data and make predictions or decisions without explicit programming."],
                      ["DLANNs imitate the human brain, enabling machines to learn from complex data for various applications."],
                      ["Natural Language Processing (NLP) enables computers to understand and process human language for various applications."]]
    g.Methodologies = zip(g.section_Method, g.section_Method_Display_Name, g.section_Info)

    # Obtain the selection of Algorithm Names and Definition.
    algorithm_Names = []
    algorithm_Definition = []
    algorithm_Names.append("")
    algorithm_Definition.append("")

    for algorithm in list_Algorithms:
        algorithm_Names.append(algorithm.getName())
        algorithm_Definition.append(algorithm.getDefinition())
    g.section_algorithm = algorithm_Names
    g.section_Info = algorithm_Definition
    g.Algorithms = zip(g.section_algorithm, g.section_Info)


    # Obtain the selection of Ensemble algorithm Names and Definition.
    Ensemble_Names = []
    Ensemble_Definition = []
    Ensemble_Names.append("")
    Ensemble_Definition.append("")

    for algorithmEnsemble in list_Ensembles:
        Ensemble_Names.append(algorithmEnsemble.getName())
        Ensemble_Definition.append(algorithmEnsemble.getDefinition())
    g.Ensemble_algorithm = Ensemble_Names
    g.Ensemble_Info = Ensemble_Definition
    g.Ensembles = zip(g.Ensemble_algorithm, g.Ensemble_Info)

    #----------------Preoptimiztion-------------------------->
    # create list of options to choose a category of preoptimization options.
    preopt_Category_Names = []
    preopt_Category_Display_Names = []
    preopt_Category_Definition = []

    for preopt_Category in list_Preopt_Categories:
        preopt_Category_Names.append(preopt_Category.getName())
        preopt_Category_Display_Names.append(preopt_Category.getDisplayName())
        preopt_Category_Definition.append(preopt_Category.getDefinition())
    g.section_preopt_Category_Names = preopt_Category_Names
    g.section_preopt_Category_Display_Names = preopt_Category_Display_Names
    g.section_preopt_Category_Info =  preopt_Category_Definition
    g.preopt_Categories = zip(g.section_preopt_Category_Names, g.section_preopt_Category_Display_Names, g.section_preopt_Category_Info)  # just the name, display names, and the definition of all

    # create list of options for the first category of preoprtimization (I.E. Standardization)
    preopt_Names = []
    preopt_Display_Names = []
    preopt_Definition = []

    for preopt in list_Standardization:
        preopt_Names.append(preopt.getName())
        preopt_Display_Names.append(preopt.getDisplayName())
        preopt_Definition.append(preopt.getDefinition())
    g.section_preopt_Names = preopt_Names
    g.section_preopt_Display_Names = preopt_Display_Names
    g.section_preopt_Info =  preopt_Definition
    g.preopts = zip(g.section_preopt_Names, g.section_preopt_Display_Names, g.section_preopt_Info)

    #-------------------Neural Network Layers----------------------->
    # create list of options to choose a category of layer options.
    layer_Category_Names = []
    layer_Category_Display_Names = []
    layer_Category_Definition = []

    for layer_Category in list_Layer_Categories:
        layer_Category_Names.append(layer_Category.getName())
        layer_Category_Display_Names.append(layer_Category.getDisplayName())
        layer_Category_Definition.append(layer_Category.getDefinition())
    g.layer_Category_Names = layer_Category_Names
    g.layer_Category_Display_Names = layer_Category_Display_Names
    g.layer_Category_Info =  layer_Category_Definition
    g.NN_layer_Categories = zip(g.layer_Category_Names, g.layer_Category_Display_Names, g.layer_Category_Info)

    # create list of options for the first category of preoprtimization (I.E. Core)
    layer_Names = []
    layer_Display_Names = []
    layer_Definition = []

    for layer in list_Core:
        layer_Names.append(layer.getName())
        layer_Display_Names.append(layer.getDisplayName())
        layer_Definition.append(layer.getDefinition())
    g.layer_Names = layer_Names
    g.layer_Display_Names = layer_Display_Names
    g.layer_Info =  layer_Definition
    g.layers = zip(g.layer_Names, g.layer_Display_Names, g.layer_Info)

    #-------------------Callbacks----------------------->
    # create list of options to choose a callback options.
    callback_Names = []
    callback_Display_Names = []
    callback_Definition = []

    for callback in list_Callbacks:
        callback_Names.append(callback.getName())
        callback_Display_Names.append(callback.getDisplayName())
        callback_Definition.append(callback.getDefinition())
    g.callback_Names = callback_Names
    g.callback_Display_Names = callback_Display_Names
    g.callback_Info =  callback_Definition
    g.callbacks = zip(g.callback_Names, g.callback_Display_Names, g.callback_Info)

@bp.route("/new_experiment")
def new_experiment():
    return render_template("experiments/new_experiment.html")

'''
To preview the LCA site
'''
@bp.route("/lca", methods=["GET"])
def lca():
    user_id = session.get("user_id")
    logging.debug("=== Uploading files: %s", user_id)

    if user_id is None:
        g.user = None
        flash("You need to login to upload your datasets!")
        return redirect(url_for("auth.login"))
    else:
        try:
            manager = DBManager.instance(
                password_file=current_app.config["DB_PASSWORD"]
            )
            plan = manager.get_plan_by_user_id(user_id)
        except Error as e:
            logging.debug(f"=== Failed to get user plan: {e} ===")
            return redirect(url_for("auth.plans"))

        logging.debug("=== Upload plan: %s", plan)
        if plan is None:
            flash("You have not subscribed to a plan!")
            return redirect(url_for("auth.plans"))
        else:
            # if plan.get('tier') == 'free':
            # TODO check file size
            user_storage = plan.get("storage_url")
            base_folder = os.getcwd()
            upload_path = Path(base_folder+'/data/') / user_storage
            filenames = os.listdir(upload_path)
            files = []
            for filename in filenames:
                file_path = upload_path / filename
                with open(file_path) as file:
                    data = csv.reader(file)
                    content = [row for row in data]
                    files.append(dict(filename=filename, content=content))

            return render_template("experiments/lca.html", files=files)


@bp.route("/cycon")
def cycon():
    user_id = session.get("user_id")

    if user_id is None:
        g.user = None
        return redirect(url_for("auth.login"))
    else:
        try:
            manager = DBManager.instance(
                password_file=current_app.config["DB_PASSWORD"]
            )
            plan = manager.get_plan_by_user_id(user_id)
        except Error:
            return redirect(url_for("auth.plans"))

        if plan is None:
            flash("You have not subscribed to a plan!")
            return redirect(url_for("auth.plans"))
        else:
            user_storage = plan.get("storage_url")
            base_folder = os.getcwd()
            upload_path = Path(base_folder+'/data/') / user_storage
            filenames = os.listdir(upload_path)
            files = []
            for filename in filenames:
                file_path = upload_path / filename
                with open(file_path) as file:
                    data = csv.reader(file)
                    content = [row for row in data]
                    files.append(dict(filename=filename, content=content))

            return render_template("experiments/cycon.html", files=files)

    return render_template("experiments/cycon.html", files=[])                                 # <------

'''
If you have a plan, returns a results as a JSON response
'''
@bp.route("/calculate", methods=["POST"])
def calculate():
    processes_input = request.form.get("processes")
    choice = request.form.get("choice")
    user_data = dict()
    user_id = session.get("user_id")

    if user_id is None:
        g.user = None
        flash("You need to login to upload your datasets!")
        return redirect(url_for("auth.login"))
    else:
        try:
            manager = DBManager.instance(
                password_file=current_app.config["DB_PASSWORD"]
            )
            plan = manager.get_plan_by_user_id(user_id)
        except Error as e:
            logging.debug(f"=== Failed to get user plan: {e} ===")
        if plan is None:
            flash('You have not yet created a plan.j')
            return redirect(url_for("auth.plans"))
        else:
            user_storage = plan.get("storage_url")
            files_path = Path('data/') / user_storage
            files = [(f.name, f.absolute())
                        for f in list(files_path.glob("*"))]
            for (key, value) in files:
                logger.debug("=== file data: %s %s", key, value)
                user_data[key], error = read_data(value)
                if error:
                    return jsonify(error=error)

    # else:
    if not processes_input:
        return flash("Invalid arguments!")
    processes = json.loads(processes_input)
    # for key, value in request.files.items():
    #     logger.debug("=== file data: %s %s", key, value)
    #     data[key], error = read_data(value)
    #     if error:
    #         return jsonify(error=error)
    logger.debug("=== Equations: %s %s", processes)

    results = []
    num_process = 0
    for process in processes:
        num_process += 1
        filename = process.get("filename")
        kind = process.get("kind")
        data = None
        if not filename:
            error = f"Missing filename for process {num_process}"
            return jsonify(error=error)
        if kind == "csv":
            data = user_data.get(filename)
        else:
            file = request.files.get(filename)
            data, error = read_data(file)
            if error:
                return jsonify(error=error)
        process_data = data

        logger.debug("=== Process data: %s", data)
        if process_data is None:
            error = f"Missing csv file '{filename}' for process {num_process}"
            return jsonify(error=error)
        equation = process.get("equation")
        if not equation or equation == "":
            error = f"Missing equation for process for process {num_process}"
            return jsonify(error=error)
        name = process.get("name")
        logger.debug("=== Process %s: %s %s",
                     num_process, filename, process_data)
        calculator = Calculator()
        evaluated, result, formula = calculator.evaluate(
            equation=equation, data=process_data)
        logger.debug("=== Process %s Results: %s, %s %s",
                     num_process, evaluated, result, formula)
        if not evaluated:
            error = f"Error in the formula for process {num_process}\n{result}"
            return jsonify(error=error)

        sympy.init_printing(use_latex='mathjax')
        formula = latex(equation, mode="plain")
        output = dict(result=result, name=name, equation=formula)
        results.append(output)

    logger.debug("=== results: %s | dtype %s", results, type(results))
    return jsonify(processes=results)

# it runs machine learning or deep learning experiments 
@bp.route("/run_experiment", methods=["POST"])
def run_experiment():    

    processes_input = request.form.get("processes")
    data = json.loads(processes_input)

    filename = request.form.get("csvFileName")
    csv_file = request.files.get("csvFile")

    data["csvFileName"] = filename
    data["csvFile"] = csv_file

    # Perform MLA if chosen
    if data['methodology'] == "MLA":
        
        # Split
        if data['validation'] == "Split":
            if 'MLalgorithm' in data and data['MLalgorithm'] != "":
                status, msg, Metrics = MLA_Validation.Split(data)
            elif 'EnsembleAlgorithm' in data and data["EnsembleAlgorithm"] != "":
                status, msg, Metrics = Ensemble_Validation.Split(data)
            

        # K-Fold
        elif data['validation'] == "K-Fold":
            if 'MLalgorithm' in data and data['MLalgorithm'] != "":
                status, msg, Metrics = MLA_Validation.K_Fold(data)
            elif "EnsembleAlgorithm" in data and data["EnsembleAlgorithm"] != "":
                status, msg, Metrics = Ensemble_Validation.K_Fold(data)


        logging.debug("statussssss %s", status)
        if status == "worked":
            # Open json file for the experiment.
            baseFolder = os.getcwd()
            locationSavedResults = Path(baseFolder) / "SavedResults"
            filename = secure_filename(data["projectName"] + ".json")
            filepath = locationSavedResults / filename.lower()
            if os.path.exists(filepath):
                os.remove(filepath)
                fp = open(filepath, "a")
            else:
                fp = open(filepath, "a")
    
            # write to json file
            metrics_Dump = json.dumps(Metrics)

            fp.write(metrics_Dump)

            # close the connection
            fp.close()

        Results = [status, msg, Metrics]

    # Perform Deep Learning Neural Network.
    if data['methodology'] == "DLANN":
        # Split
    
        status, msg, Metrics = DLANN_Validation.Split(data)
        
        if status == "worked":
            # Open json file for the experiment.
            baseFolder = os.getcwd()
            locationSavedResults = Path(baseFolder) / "SavedResults"
            filename = secure_filename(data["projectName"] + ".json")
            filepath = locationSavedResults / filename.lower()
            if os.path.exists(filepath):
                os.remove(filepath)
                fp = open(filepath, "a")
            else:
                fp = open(filepath, "a")
    
            # write to json file
            metrics_Dump = json.dumps(Metrics)

            fp.write(metrics_Dump)

            # close the connection
            fp.close()

        Results = [status, msg, Metrics]

    return json.dumps(Results)


@bp.route("/getResults", methods=["POST"])
def getResults(): 
    output = request.get_json()
    formData = json.loads(output)

    data = formData['form']

    baseFolder = os.getcwd()
    locationSavedResults = Path(baseFolder) / "SavedResults"
    filename = data["projectName"] + ".json"
    filepath = locationSavedResults / filename.lower()
    fp = open(filepath, "r")

    Metrics = json.load(fp)

    # close the connection
    fp.close()
    
    return json.dumps(Metrics)

# get ensemble method algorithm parameeters
@bp.route("/getEnsembleParameters", methods=["POST"])
def getEnsembleParameters():
    output = request.get_json()
    data = json.loads(output)

    Parameters = Ensemble.getParameters(data["Algorithm"])

    return json.dumps(Parameters)


# Get machine learning algorithms parameters for the algorithm in data that we created in cycon_events
@bp.route("/getAlgorithmParameters", methods=["POST"])
def getAlgorithmParameters():
    output = request.get_json()
    data = json.loads(output)

    Parameters = MLA.getParameters(data["Algorithm"])

    return json.dumps(Parameters)
   
   # Definition ::
   # As an Inputs it recieves catagoryName 

   # Output the optimizer in json format
@bp.route("/getCategoryPreopts", methods=["POST"])
def getCategoryPreopts():
    output = request.get_json() # Recieved {catagory: catagoryName}
    data = json.loads(output)

    preopts = Preoptimization.getCategoryPreopts(data["Category"])  # send the catagoryName as value and the getCataforyPropts() returns the optimizer for that catagory
                                                                    # it includes Name, Display_Name, Definition, Parameters
    return json.dumps(preopts)

  # Definition ::

   # Inputs it recieves catagoryName 

   # Output the layer in json format
@bp.route("/getCategoryLayers", methods=["POST"])
def getCategoryLayers():
    output = request.get_json()
    data = json.loads(output)

    layers = NeuralNetwork.getCategoryLayers(data["Category"])

    return json.dumps(layers)



   # Output the result of rendering the HTML template
@bp.route("/results")
def results():
    return render_template("experiments/results.html")



# Inputs contains process, CSVfileName, CSVfile
   # Output the result in PDF
# @bp.route("/getTextResults", methods=["POST"]) 
# def getTextResults():
#     # the request contains process, CSVfileName, CSVfile
    
#     status, msg, info = Preoptimization.getCSV_PDF(data)

#     Results = [status, msg, info]

#     return json.dumps(Results)




   # Inputs contains process, CSVfileName, CSVfile
   # Output the result in PDF
@bp.route("/getCSVResults", methods=["POST"]) 
def getCSVResults():
    # the request contains process, CSVfileName, CSVfile
    processes_input = request.form.get("processes")
    data = json.loads(processes_input)

    filename = request.form.get("csvFileName")
    csv_file = request.files.get("csvFile")

    data["csvFileName"] = filename
    data["csvFile"] = csv_file

    status, msg, info = Preoptimization.getCSV_PDF(data)

    Results = [status, msg, info]

    return json.dumps(Results)


   # Inputs the process involved while creating the model
   # Output the summary of the model
@bp.route("/getModelSummary", methods=["POST"])
def getModelSummary():
    processes_input = request.form.get("processes")
    data = json.loads(processes_input)

    status, msg, info = NeuralNetwork.getModelSummary(data)

    Results = [status, msg, info]

    return json.dumps(Results)

   # Inputs the preprocess involved in the csv
   # Output the csv after preprocess
@bp.route("/downloadCSV", methods=["POST"])
def downloadCSV():
    processes_input = request.form.get("processes")
    data = json.loads(processes_input)

    filename = request.form.get("csvFileName")
    csv_file = request.files.get("csvFile")

    data["csvFileName"] = filename
    data["csvFile"] = csv_file

    status, msg, info = Preoptimization.downloadCSV(data)

    Results = [status, msg, info]

    return json.dumps(Results)

# Inputs the preoptimization name
# Output the parameters give the name in json formate
@bp.route("/getPreoptParameters", methods=["POST"])
def getPreoptParameters(): # sending preoptimization name
    output = request.get_json()
    data = json.loads(output)

    Parameters = Preoptimization.getParameters(data["Preopt"]) # gets the parameters given the name

    return json.dumps(Parameters)

# Inputs the data
# Output the parameter of the layer in json format
@bp.route("/getLayerParameters", methods=["POST"])
def getLayerParameters():
    output = request.get_json()
    data = json.loads(output)        # This line parses the JSON data retrieved from the request and converts it into a Python dictionary. The json.loads() function is used to deserialize the JSON string.

    Parameters = NeuralNetwork.getParameters(data["Layer"])

    return json.dumps(Parameters)

# Inputs the data
# Output the parameter of the callback in json format
@bp.route("/getCallbackParameters", methods=["Post"])
def getCallbackParameters():
    output = request.get_json()
    data = json.loads(output)

    Parameters = Callbacks.getParameters(data["Callback"])

    return json.dumps(Parameters)

# Gets the colummn names inside the csv.
@bp.route("/getCSVColumnTitles", methods=["POST"])
def getCSVColumnTitles():
    filename = request.form.get("csvFileName")
    csv_file = request.files.get("csvFile")

    data = {"csvFileName": filename, "csvFile": csv_file}

    columnTitles = Preoptimization.getCSVColumnTitles(data)

    return json.dumps(columnTitles)

# Gets all the options that is available via the keras model.compile 
@bp.route("/getCompilerOptions", methods=["POST"])
def getCompilerOptions():
    compiler_Options = Compiler.getCompilerOptions()

    return json.dumps(compiler_Options)

# Gets all the options that is available for the model to fit,train, and validate.
@bp.route("/getValidationOptions", methods=["POST"])
def getValidationOptions():
    validation_Options = NN_Validation.getValidationOptions()

    return json.dumps(validation_Options)