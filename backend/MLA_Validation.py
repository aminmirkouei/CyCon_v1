import os
import sys
import io
import base64

from backend import MLA

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path
from werkzeug.utils import secure_filename

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from backend.Classes.PreoptimizationFiles import Preoptimization
import traceback
import logging
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from math import sqrt
import traceback


def getDataset(fileName):
     # convert csv to usable dataset
    ## Manual location currently...Will be changed when implemented in Host.
    ## Should further be changed when database is setup.
    baseFolder = os.getcwd()

    ManualLoc = Path(baseFolder) / "tests/sampleCSV_MLA_Classification/"

    filename = secure_filename(fileName)
    dataset_path = ManualLoc / filename.lower()
    dataset = pd.read_csv(dataset_path)

    return dataset

def Split(data):
    try:
        # convert csv to usable dataset
        ##df_act = getDataset(data['csvFileName'])

        if data['csvFileName'].endswith('.npy'):
            numpy_array = np.load(data['csvFile'])
            df_act = pd.DataFrame(data=numpy_array[1:, :], columns=numpy_array[0, :])
        elif data['csvFileName'].endswith('.csv'):
            df_act = pd.read_csv(data['csvFile'], index_col=None)
        
        # logging.debug(df_act)
        # df_act = pd.read_csv(data['csvFile'], index_col=None)
       
        # Cycle through the choices of preotimization
        for i in range(int(data["preoptCounter"])):
            # Get the choice of preoptimization.
            preopt_method = data["Preopt_" + str(i)]
            # Perform the preoptimization on the dataset.
            df_act = Preoptimization.perform_Preopt(data, i, df_act)

        
        X = df_act.drop(data["class_col"], axis=1)  # Features
        y = df_act[data["class_col"]] 

        # df = df_act.to_numpy()
        # y = df[:, -1]
      
        # logging.debug("Testing_11111111111")
        # Split dataset to training and testing set
        random_state = None
        if data["Random_State_Input"] != "":
            random_state = int(data["Random_State_Input"])
        shuffle = False
        if data["Shuffle"] == "True":
            shuffle = True
        stratify = None
        if data["Stratify"] == "True":
            stratify = df_act[data["class_col"]]
        
        if data[data['validation'] + "_Input"] == "0.0":
            x_train = X
        
            y_train = y.to_numpy()
            model, settings = MLA.createModel(data)
            # Perform the Method.
            model.fit(x_train, y_train)
            weight = "weight not available"
            try:
                
                if data['MLalgorithm'] == "RandomForestClassifier" or data['MLalgorithm'] == "RandomForestRegression":
                    weight = model.feature_importances_
                elif data['MLalgorithm'] == "SVR":
                    dual_coefficients = model.dual_coef_
                    support_vectors = model.support_vectors_
                    weights = np.abs(dual_coefficients) @ support_vectors
                    weight = weights*0.73
                    
                elif hasattr(model, 'coef_'):
                    weight = model.coef_
                else:
                    weight = "weight not available"
            except Exception as e:
                weight = "weight not available"
                
            logging.debug(data['MLalgorithm'] )
            Accuracy = "This metric needs split more than 0.0"
            F1 = "This metric needs split more than 0.0"
            F1_micro = "This metric needs split more than 0.0"
            F1_macro = "This metric needs split more than 0.0"
            Precision = "This metric needs split more than 0.0"
            Precision_micro = "This metric needs split more than 0.0"
            Precision_macro = "This metric needs split more than 0.0"
            recall = "This metric needs split more than 0.0"
            recall_micro = "This metric needs split more than 0.0"
            recall_macro = "This metric needs split more than 0.0"
            my_base64_jpgData = "This metric needs split more than 0.0"

            mse = "This metric needs split more than 0.0"
            rmse = "This metric needs split more than 0.0"
            mae = "This metric needs split more than 0.0"
            r2 = "This metric needs split more than 0.0"

        else:
            # train_set, test_set = train_test_split(df, test_size=float(data[data['validation'] + "_Input"]), shuffle=shuffle, random_state = random_state, stratify = stratify)
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=float(data[data['validation'] + "_Input"]), shuffle=shuffle, random_state = random_state, stratify = stratify)
            # length = train_set.shape[1] -1
        
            # x_train = train_set[:,0:length]
            # y_train = train_set[:,length]
            # x_test = test_set[:,0:length]
            # y_test = test_set[:,length]
            # logging.debug("Testing_6666666")
            #model = KNeighborsClassifier(n_neighbors=3)
            model, settings = MLA.createModel(data)
        
            
            # Perform the Method.
            model.fit(x_train, y_train)
            
            # Predict the testset
            y_pred = model.predict(x_test)
            Accuracy = "This metric is for classification"
            F1 = "This metric is for classification"
            F1_micro = "This metric is for classification"
            F1_macro = "This metric is for classification"
            Precision = "This metric is for classification"
            Precision_micro = "This metric is for classification"
            Precision_macro = "This metric is for classification"
            recall = "This metric is for classification"
            recall_micro = "This metric is for classification"
            recall_macro = "This metric is for classification"
            my_base64_jpgData = "This metric is for classification"

            mse = "This metric is for Regression"
            rmse = "This metric is for Regression"
            mae = "This metric is for Regression"
            r2 = "This metric is for Regression"
            # Obtain the Metrics
            if data["regression"] == "false":
                Accuracy = accuracy_score(y_test, y_pred)
                F1 = f1_score(y_test, y_pred, average=None)
                F1 = F1.tolist()
                F1_micro = f1_score(y_test, y_pred, average='micro')
                F1_macro = f1_score(y_test, y_pred, average='macro')
                Precision = precision_score(y_test, y_pred, average=None)
                Precision = Precision.tolist()
                Precision_micro = precision_score(y_test, y_pred, average='micro')
                Precision_macro = precision_score(y_test, y_pred, average='macro')
                recall = recall_score(y_test, y_pred, average=None)
                recall_micro = recall_score(y_test, y_pred, average='micro')
                recall_macro = recall_score(y_test, y_pred, average='macro')
                recall = recall.tolist()
                confusion_matrix(y_test, y_pred)

                cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
                color = 'white'
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
                disp.plot()  # generates the plot of the confusion matrix using matplotlib.
                my_stringIObytes = io.BytesIO()  # creates an in-memory binary stream to store the plot image.
                plt.savefig(my_stringIObytes, format='jpg')  # saves the plot image to the my_stringIObytes stream in JPEG format.
                plt.close()
                my_stringIObytes.seek(0) # moves the stream's position to the beginning, preparing it for reading.
                my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode() # reads the content of the stream, encodes it in Base64 format, and converts it to a string.
                if data['MLalgorithm'] == "RandomForestClassifier" or data['MLalgorithm'] == "RandomForestRegression":
                    weight = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    weight = model.coef_
                else:
                    weight = "weight not available"

            else:
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                if data['MLalgorithm'] == "RandomForestClassifier" or data['MLalgorithm'] == "RandomForestRegression":
                    weight = model.feature_importances_
                elif data['MLalgorithm'] == "SVR":
                    dual_coefficients = model.dual_coef_
                    support_vectors = model.support_vectors_
                    weights = np.abs(dual_coefficients) @ support_vectors
                    weight = weights*0.73
                elif hasattr(model, 'coef_'):
                    weight = model.coef_
                else:
                    weight = "weight not available"
                
            # create a confusion matrix
            #def fig_to_base64(fig):
            #    img = io.BytesIO()
            #    fig.savefig(img, format='png',
            #            bbox_inches='tight')
            #    img.seek(0)

            #    return base64.b64encode(img.getvalue())

        
            #temp = os.getcwd()
            #plt.savefig(temp + '\\backend\\static\\Images\\' + data['projectName'] + '.png')

            #data_uri = base64.b64encode(open('conf_matrix.png', 'rb').read()).decode('utf-8')
            #img_tag = '<img src="data:image/png;base64,{0}">'.format(data_uri)

            #encoded = fig_to_base64(fig)
            #my_html = '<img src="data:image/png;base64, {}">'.format(encoded.decode('utf-8'))

            # logging.debug("Testing_3333333333")
            # Send the Metrics
        Metrics = {"Validation": "Split",

                   "Accuracy_Intro": 'Accuracy: ',
                   "Precision_Intro": "Precision for Each Class: ",
                   "Precision_micro_Intro": "Precision (Micro): ",
                   "Precision_macro_Intro": "Precision (Macro): ",
                   "F1_Intro": "F1 for each Class: ",
                   "F1_micro_Intro": "F1 (Micro): ",
                   "F1_macro_Intro": "F1 (Macro): ",
                   "Recall_Intro": "Recall for each class: ",
                   "Recall_micro_Intro": "Recall (Micro): ",
                   "Recall_macro_Intro": "Recall (Macro): ",

                    "Accuracy": Accuracy,
                    "Precision": Precision,
                    "Precision_micro": Precision_micro,
                    "Precision_macro": Precision_macro,
                    "F1": F1,
                    "F1_micro": F1_micro,
                    "F1_macro": F1_macro,
                    "recall" : recall,
                    "recall_macro": recall_macro,
                    "recall_micro": recall_micro,
                    "weights": str(weight),

                    "cm_overall": my_base64_jpgData,
                    "MSE": mse,
                    "RMSE": rmse,
                    "MAE": mae,
                    "r2": r2,
                
                    "Val_Random_State": random_state,
                    "Val_Shuffle": shuffle}

        Metrics.update(settings)

        status = "worked"
        msg = ""

        return status, msg, Metrics

    except Exception as e:
        line_number = traceback.extract_tb(e.__traceback__)[-1].lineno
        Metrics = ""
        msg = f"Error occurred at: {str(e)}"
        status = "error"

        return status, msg, Metrics


def K_Fold(data):
    try:
        # convert csv to usable dataset
        ##df_act = getDataset(data['csvFileName'])
        if data['csvFileName'].endswith('.npy'):
            numpy_array = np.load(data['csvFile'])
            df_act = pd.DataFrame(data=numpy_array[1:, :], columns=numpy_array[0, :])
        elif data['csvFileName'].endswith('.csv'):
            df_act = pd.read_csv(data['csvFile'], index_col=None)
        
        # df_act = pd.read_csv(data['csvFile'], index_col=None)
        # logging.debug(df_act)
        # Cycle through the choices of preotimization
        for i in range(int(data["preoptCounter"])):
            # Get the choice of preoptimization.
            preopt_method = data["Preopt_" + str(i)]
            # Perform the preoptimization on the dataset.
            df_act = Preoptimization.perform_Preopt(data, i, df_act)

        
        df = df_act.to_numpy()
        

        length = df.shape[1] -1

        # X = df[:,0:length]
        # y = df[:,length]


        X = df_act.drop(data["class_col"], axis=1).values  # Features
        y = df_act[data["class_col"]].values
        # x_train = X
        
            # y_train = y.to_numpy() 

        random_state = None
        if data["Random_State_Input"] != "":
            random_state = int(data["Random_State_Input"])
        shuffle = False
        if data["Shuffle"] == "True":
            shuffle = True

        if data["Stratify"] == "False":
            kf = KFold(n_splits=int(data[data['validation'] + "_Input"]), shuffle=shuffle, random_state = random_state)
        if data["Stratify"] == "True":
            kf = StratifiedKFold(n_splits=int(data[data['validation'] + "_Input"]), shuffle=shuffle, random_state = random_state)
        
        number_of_kfold = int(data[data['validation'] + "_Input"])
        kf.get_n_splits(X)

        if data["regression"] == "false":
            acc_list = []
            prec_list = []
            prec_micro_list = []
            prec_macro_list = []
            f1_list = []
            f1_micro_list = []
            f1_macro_list = []
            recall_list = []
            recall_micro_list = []
            recall_macro_list = []
            cm_list = []
            y_test_list = np.empty(1)
            y_predict_list = np.empty(1)


            mse = ["This is for regression"]
            rmse = ["This is for regression"]
            mae = ["This is for regression"]
            r2 = ["This is for regression"]



            mean_mse = ["This is for regression"]
            mean_rmse = ["This is for regression"]
            mean_mae = ["This is for regression"]
            mean_r2 = ["This is for regression"]
        

            for i, (train_index, test_index) in enumerate(kf.split(X,y)):
                print("Fold: " + str(i) + " ===============================")
                x_train = X[train_index]
                y_train = y[train_index]
                x_test = X[test_index]
                y_test = y[test_index]
                
                model, settings = MLA.createModel(data)

                model.fit(X[train_index], y[train_index])
        
                y_pred = model.predict(X[test_index])

                acc = accuracy_score(y[test_index], y_pred)
                prec = precision_score(y[test_index], y_pred, average=None)
                prec_micro = precision_score(y[test_index], y_pred, average='micro')
                prec_macro = precision_score(y[test_index], y_pred, average='macro')
                f1 = f1_score(y[test_index], y_pred, average=None)
                f1_micro = f1_score(y[test_index], y_pred, average='micro')
                f1_macro = f1_score(y[test_index], y_pred, average='macro')

                recall = recall_score(y[test_index], y_pred, average=None)
                recall_micro = recall_score(y[test_index], y_pred, average='micro')
                recall_macro = recall_score(y[test_index], y_pred, average='macro')
        
                acc_list.append(acc)
                prec_list.append(prec)
                prec_micro_list.append(prec_micro)
                prec_macro_list.append(prec_macro)
                f1_list.append(f1)
                f1_micro_list.append(f1_micro)
                f1_macro_list.append(f1_macro)

                recall_list.append(recall)
                recall_micro_list.append(recall_micro)
                recall_macro_list.append(recall_macro)

                y_test_list = np.concatenate((y_test_list, y[test_index]))
                y_predict_list = np.concatenate((y_predict_list, y_pred))
        
                cm = confusion_matrix(y[test_index], y_pred, labels=model.classes_)
                color = 'white'
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
                disp.plot()
                my_stringIObytes = io.BytesIO()
                plt.savefig(my_stringIObytes, format='jpg')
                my_stringIObytes.seek(0)
                my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode()
                cm_list.append(my_base64_jpgData)

            
            acc_list = np.array(acc_list)
            acc_list = acc_list.tolist()
            prec_list = np.array(prec_list)
            prec_list = prec_list.tolist()
            prec_micro_list = np.array(prec_micro_list)
            prec_micro_list = prec_micro_list.tolist()
            prec_macro_list = np.array(prec_macro_list)
            prec_macro_list = prec_macro_list.tolist()
            f1_list = np.array(f1_list)
            f1_list = f1_list.tolist()
            f1_micro_list = np.array(f1_micro_list)
            f1_micro_list = f1_micro_list.tolist()
            f1_macro_list = np.array(f1_macro_list)
            f1_macro_list = f1_macro_list.tolist()

            recall_list = np.array(recall_list)
            recall_list = recall_list.tolist()
            recall_micro_list = np.array(recall_micro_list)
            recall_micro_list = recall_micro_list.tolist()
            recall_macro_list = np.array(recall_macro_list)
            recall_macro_list = recall_macro_list.tolist()

            cm_list = np.array(cm_list)
            cm_list = cm_list.tolist()

            y_test_list = np.delete(y_test_list, 0)
            y_predict_list = np.delete(y_predict_list, 0)

            acc_average = np.average(acc_list)
            prec_average = np.average(prec_list, axis = 0)
            prec_average = prec_average.tolist()
            prec_micro_average = np.average(prec_micro_list)
            prec_macro_average = np.average(prec_macro_list)
            f1_average = np.average(f1_list, axis = 0)
            f1_average = f1_average.tolist()
            f1_micro_average = np.average(f1_micro_list)
            f1_macro_average = np.average(f1_macro_list)

            recall_average = np.average(recall_list, axis = 0)
            recall_average = recall_average.tolist()
            recall_micro_average = np.average(recall_micro_list)
            recall_macro_average = np.average(recall_macro_list)

            cm = confusion_matrix(y_test_list, y_predict_list, labels=model.classes_)
            color = 'white'
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
            disp.plot()
            my_stringIObytes = io.BytesIO()
            plt.savefig(my_stringIObytes, format='jpg')
            my_stringIObytes.seek(0)
            my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode()

        else:
            acc_list = ["This is for classification"]
            prec_list = ["This is for classification"]
            prec_micro_list = ["This is for classification"]
            prec_macro_list = ["This is for classification"]
            f1_list = ["This is for classification"]
            f1_micro_list = ["This is for classification"]
            f1_macro_list = ["This is for classification"]
            recall_list = ["This is for classification"]
            recall_micro_list = ["This is for classification"]
            recall_macro_list = ["This is for classification"]
            cm_list = ["This is for classification"]
            acc_average = ["This is for classification"]
            prec_average = ["This is for classification"]
            prec_micro_average = ["This is for classification"]
            prec_macro_average = ["This is for classification"]
            f1_average = ["This is for classification"]
            f1_micro_average = ["This is for classification"]
            f1_macro_average = ["This is for classification"]
            recall_average = ["This is for classification"]
            recall_micro_average = ["This is for classification"]
            recall_macro_average = ["This is for classification"]
            my_base64_jpgData = ["This is for classification"]

            

            mse = []
            rmse = []
            mae = []
            r2 = []

            # for i, (train_index, test_index) in enumerate(kf.split(X,y)):
            #     print("Fold: " + str(i) + " ===============================")
            #     x_train = X[train_index]
            #     y_train = y[train_index]
            #     x_test = X[test_index]
            #     y_test = y[test_index]

            # Perform K-Fold cross-validation
            for i, (train_index, test_index) in enumerate(kf.split(X,y)):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model, settings = MLA.createModel(data)

                model.fit(X[train_index], y[train_index])
        
                y_pred = model.predict(X[test_index])

                # Calculate evaluation metrics
                mse_scores = mean_squared_error(y_test, y_pred)
                rmse_score = sqrt(mse_scores)
                mae_scores = mean_absolute_error(y_test, y_pred)
                r2_scores = r2_score(y_test, y_pred)


                # Append scores to lists
                mse.append(mse_scores)
                rmse.append(rmse_score)
                mae.append(mae_scores)
                r2.append(r2_scores)
            
            # Calculate the mean and standard deviation of the scores
            mean_mse = np.mean(mse)
            mean_rmse = np.mean(rmse)
            mean_mae = np.mean(mae)
            mean_r2 = np.mean(r2)

            std_mse = np.std(mse)
            
            std_mae = np.std(mae)
            mean_r2 = np.mean(r2)
            std_r2 = np.std(r2)

        logging.debug("This the output")
            

        # Send the Metrics
        Metrics = {"Validation": "K-Fold",
                   "Number_K_fold": number_of_kfold,
                   "Accuracy_Intro": 'Accuracy: ',
                   "Precision_Intro": "Precision for Each Class: ",
                   "Precision_micro_Intro": "Precision (Micro): ",
                   "Precision_macro_Intro": "Precision (Macro): ",
                   "F1_Intro": "F1 for each Class: ",
                   "F1_micro_Intro": "F1 (Micro): ",
                   "F1_macro_Intro": "F1 (Macro): ",
                   "Recall_Intro": "Recall for each class: ",
                   "Recall_micro_Intro": "Recall (Micro): ",
                   "Recall_macro_Intro": "Recall (Macro): ",

                   "Accuracy_Intro_Overall": 'Average Accuracy: ',
                   "Precision_Intro_Overall": "Average Precision for Each Class: ",
                   "Precision_micro_Intro_Overall": "Average Precision (Micro): ",
                   "Precision_macro_Intro_Overall": "Average Precision (Macro): ",
                   "F1_Intro_Overall": "Average F1 for each Class: ",
                   "F1_micro_Intro_Overall": "Average F1 (Micro): ",
                   "F1_macro_Intro_Overall": "Average F1 (Macro): ",
                   "Recall_Intro_Overall": "Average Recall for each class: ",
                   "Recall_micro_Intro_Overall": "Average Recall (Micro): ",
                   "Recall_macro_Intro_Overall": "Average Recall (Macro): ",

                    "acc_list": acc_list, 
                    "prec_list": prec_list,
                    "prec_micro_list": prec_micro_list,
                    "prec_macro_list": prec_macro_list,
                    "f1_list": f1_list,
                    "f1_micro_list": f1_micro_list,
                    "f1_macro_list": f1_macro_list,
                    "recall_list": recall_list,
                    "recall_micro_list": recall_micro_list,
                    "recall_macro_list": recall_macro_list,
                    "cm_list": cm_list,

                    "acc_average": acc_average,
                    "prec_average": prec_average,
                    "prec_micro_average": prec_micro_average,
                    "prec_macro_average": prec_macro_average,
                    "f1_average": f1_average,
                    "f1_micro_average": f1_micro_average,
                    "f1_macro_average": f1_macro_average,
                    "recall_average": recall_average,
                    "recall_micro_average": recall_micro_average,
                    "recall_macro_average": recall_macro_average,
                    "cm_overall": my_base64_jpgData,

                    
                    "MSE": mse,
                    "RMSE": rmse,
                    "MAE": mae,
                    "r2": r2,

                    "mean_mse": mean_mse,
                    "mean_rmse": mean_rmse,
                    "mean_mae": mean_mae,
                    "mean_r2": mean_r2,
                    
                

                
                    "Val_Random_State": random_state,
                    "Val_Shuffle": shuffle}

        Metrics.update(settings)

        status = "worked"
        msg = ""

        return status, msg, Metrics

    except Exception as e:
        Metrics = ""
        msg = str(e)
        
        status = "error"

        error_message = str(e)
        error_type = type(e).__name__
        error_traceback = traceback.format_exc()
        logging.debug(error_traceback, "ERRRRRRRROR")
        # Print the error message with traceback
        # print(f"An error of type '{error_type}' occurred: {error_message}\n{error_traceback}")

        return status, msg, Metrics

