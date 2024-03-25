
# File contains all information pertaining to the Machine Learning Algortihms.

# Classification
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

class Ensemble:
    def __init__(self, name, definition, parameters):
        self.name = name
        self.definition = definition
        self.parameters = parameters
    
    def getName(self):
        return self.name

    def getDefinition(self):
        return self.definition

    def getParameters(self):
        return self.parameters

list_Ensembles = []


Name = "XGBClassifier"
Definition = ["KXGBClassifier info"]
Parameter_0 = {"Name":"n_estimators", 
               "Type": ["int_or_null"], 
               "Default_option":None,
               "Default_value":None,
               "Possible":["int"], 
               "Definition":"Number of boosting rounds."
               }
Parameter_1 = {"Name":"max_depth", 
               "Type": ["int"], 
               "Default_option":6,
               "Default_value":6,
               "Possible":["int"], 
               "Definition":"Maximum tree depth for base learners."
               }

Parameter_2 = {"Name":"max_leaves", 
               "Type": ["int"], 
               "Default_option":0,
               "Default_value":0,
               "Possible":["int"], 
               "Definition":"Maximum number of leaves; 0 indicates no limit."
               }
Parameter_3 = {"Name":"learning_rate", 
               "Type": ["float"], 
               "Default_option":0.3, 
               "Default_value":0.3, 
               "Possible":["float"],
               "Definition":"Step size shrinkage used in update to prevents overfitting."
               }
Parameter_4 =  {"Name":"objective", "Type": ["option"], "Default_option":"reg:squarederror", "Default_value":"reg:squarederror", "Possible":["reg:squarederror", "binary:logistic", "multi:softmax", "rank:pairwise", "rank:ndcg", "count:poisson", "survival:cox"], 
               "Definition":"Specify the learning task and the corresponding learning objective"}

Parameter_5 = {"Name":"booster", "Type": ["option"], "Default_option":"gbtree", "Default_value":"gbtree", "Possible":["gbtree","gblinear","dart"], 
                "Definition":"Specify which booster to use"}

# Parameter_1 = {"Name":"weights", "Type": ["option"], "Default_option":"uniform", "Default_value":"uniform", "Possible":["uniform","distance"], 
#                 "Definition":"Weight function used in prediction. Possible values:\n\n‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.\n‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.\n[callable] : a user-defined function which accepts an array of distances, and returns an array of the same shape containing the weights."}
# Parameter_2 = {"Name":"algorithm", "Type": ["option"], "Default_option":"auto", "Default_value":"auto", "Possible":["ball_tree","kd_tree","brute","auto"],
#               "Definition":"Algorithm used to compute the nearest neighbors:\n\n'ball_tree’ will use BallTree\n‘kd_tree’ will use KDTree\n‘brute’ will use a brute-force search.\n‘auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit method."}
# Parameter_3 = {"Name":"leaf_size", "Type": ["int"], "Default_option":30, "Default_value":30, "Possible":["int"],
#               "Definition":"Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem."}
# Parameter_4 = {"Name":"p", "Type": ["int"], "Default_option":2, "Default_value":2, "Possible":["int"],
#               "Definition":"Power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used."}
Parameters = {"Parameter_0":Parameter_0, "Parameter_1":Parameter_1, "Parameter_2":Parameter_2, "Parameter_3":Parameter_3, "Parameter_4": Parameter_4, "Parameter_5": Parameter_5}

# Parameters = {"Parameter_0":Parameter_0, "Parameter_1":Parameter_1,"Parameter_2":Parameter_2,"Parameter_3":Parameter_3,
#                   "Parameter_4":Parameter_4}

XGBClassifier_algorithm = Ensemble(Name, Definition, Parameters)

list_Ensembles.append(XGBClassifier_algorithm)

# LGBMClassifier
Name = "LGBMClassifier"
Definition = ["LightGBM gradient boosting framework for classification tasks."]
Parameter_0 = {"Name":"boosting_type", "Type": ["option"], "Default_option":"gbdt", "Default_value":"gbdt", "Possible":["gbdt","dart","goss"], 
                "Definition":"The boosting type to be used"}
Parameter_1 = {"Name":"num_leaves", "Type": ["int"], "Default_option":31, "Default_value":31, "Possible":["int"],
              "Definition":"The maximum number of leaves in one tree."}

Parameter_2 = {"Name":"learning_rate", "Type": ["float"], "Default_option":0.1, "Default_value":0.1, "Possible":["float"], 
               "Definition":"The learning rate or shrinkage rate for boosting."}
Parameter_3 = {"Name":"n_estimators", "Type": ["int"], "Default_option":100, "Default_value":100, "Possible":["int"],
              "Definition":"The number of boosting iterations or the number of trees to build."}
Parameter_4 = {"Name":"objective", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":["None","binary","multiclass"], 
                "Definition":"The objective function to be optimized."}
Parameter_5 = {"Name":"reg_alpha", "Type": ["float"], "Default_option":0.0, "Default_value":0.0, "Possible":["float"], 
               "Definition":"L1 regularization term on weights. "}

Parameter_6 = {"Name":"reg_lambda", "Type": ["float"], "Default_option":0.0, "Default_value":0.0, "Possible":["float"], 
               "Definition":"L2 regularization term on weights."}

Parameters = {"Parameter_0":Parameter_0, "Parameter_1":Parameter_1,"Parameter_2":Parameter_2,"Parameter_3":Parameter_3,
             "Parameter_4":Parameter_4,"Parameter_5":Parameter_5,"Parameter_6":Parameter_6}

LGBMClassifier_algorithm = Ensemble(Name, Definition, Parameters)
list_Ensembles.append(LGBMClassifier_algorithm)



# CatBoost
Name = "CatBoost"
Definition = ["CatBoost is a machine learning algorithm that uses gradient boosting on decision trees."]
Parameter_0 = {"Name":"loss_function", "Type": ["option"], "Default_option":"Logloss", "Default_value":"Logloss", "Possible":["Logloss","MultiClass","CrossEntropy","MAE"], 
                "Definition":"Determines the metric used during training and the machine learning problem to solve."}
Parameter_1 = {"Name":"iterations", "Type": ["int"], "Default_option":100, "Default_value":100, "Possible":["int"],
              "Definition":"The maximum number of boosting iterations to perform."}

Parameter_2 = {"Name":"learning_rate", "Type": ["float"], "Default_option":0.03, "Default_value":0.03, "Possible":["float"], 
               "Definition":"The step size shrinkage used in update to prevent overfitting. Lower values make the model more robust but require more iterations."}

Parameter_3 = {"Name":"depth", "Type": ["int"], "Default_option":6, "Default_value":6, "Possible":["int"],
              "Definition":"The depth of the trees in the model. A higher value can lead to more complex models but also increases the risk of overfitting."}

Parameter_4 = {"Name":"l2_leaf_reg", "Type": ["float"], "Default_option":3.0, "Default_value":3.0, "Possible":["float"], 
               "Definition":"Coefficient at the L2 regularization term of the cost function. It helps to avoid overfitting."}

Parameter_5 = {"Name":"random_strength", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
              "Definition":"Controls the level of randomness in the model."}

Parameter_6 = {"Name":"border_count", "Type": ["int"], "Default_option":254, "Default_value":254, "Possible":["int"],
              "Definition":"Specifies the number of splits for numerical features."}

Parameter_7 = {"Name":"bagging_temperature", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
              "Definition":"Controls the strength of the Bayesian bagging. "}


Parameters = {"Parameter_0":Parameter_0, "Parameter_1":Parameter_1,"Parameter_2":Parameter_2,"Parameter_3":Parameter_3,
             "Parameter_4":Parameter_4, "Parameter_5":Parameter_5, "Parameter_6":Parameter_6, "Parameter_7":Parameter_7}

CatBoost_algorithm = Ensemble(Name, Definition, Parameters)
list_Ensembles.append(CatBoost_algorithm)
# svm_algorithm = MLA(Name, Definition, Parameters)

# list_MLAs.append(svm_algorithm)

# # Decision Trees
# Name = "DecisionTreeClassifier"
# Definition = ["A decision tree classifier."]
# Parameter_0 =  {"Name":"criterion", "Type": ["option"], "Default_option":"gini", "Default_value":"gini", "Possible":["gini", "entropy", "log_loss"], 
#                "Definition":"The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “log_loss” and “entropy” both for the Shannon information gain."}
# Parameter_1 =  {"Name":"splitter", "Type": ["option"], "Default_option":"best", "Default_value":"best", "Possible":["best","random"], 
#                "Definition":"The strategy used to choose the split at each node. Supported strategies are “best” to choose the best split and “random” to choose the best random split."}
# Parameter_2 =  {"Name":"max_depth", "Type": ["int_or_null"], "Default_option":None, "Default_value":None, "Possible":["int"], 
#                "Definition":"The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples."}
# Parameter_3 =  {"Name":"min_samples_split", "Type": ["int"], "Default_option":2, "Default_value":2, "Possible":["int"], 
#                "Definition":"The minimum number of samples required to split an internal node:\n\nIf int, then consider min_samples_split as the minimum number.\n\nIf float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split."}
# Parameter_4 =  {"Name":"min_samples_leaf", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"], 
#                "Definition":"The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.\n\nIf int, then consider min_samples_leaf as the minimum number.\n\nIf float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node."}
# Parameter_5 =  {"Name":"min_weight_fraction_leaf", "Type": ["float"], "Default_option":0.0, "Default_value":0.0, "Possible":["float"], 
#                "Definition":"The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided."}
# Parameter_6 =  {"Name":"max_features", "Type": ["int_or_null"], "Default_option":None, "Default_value":None, "Possible":["int"], 
#                "Definition":"The number of features to consider when looking for the best split:\n\nIf int, then consider max_features features at each split.\n\nIf float, then max_features is a fraction and max(1, int(max_features * n_features_in_)) features are considered at each split.\n\nIf “auto”, then max_features=sqrt(n_features).\n\nIf “sqrt”, then max_features=sqrt(n_features).\n\nIf “log2”, then max_features=log2(n_features).\n\nIf None, then max_features=n_features."}
# Parameter_7 =  {"Name":"random_state", "Type": ["int_or_null"], "Default_option":None, "Default_value":None, "Possible":["int"], 
#                "Definition":'Controls the randomness of the estimator. The features are always randomly permuted at each split, even if splitter is set to "best". When max_features < n_features, the algorithm will select max_features at random at each split before finding the best split among them. But the best found split may vary across different runs, even if max_features=n_features. That is the case, if the improvement of the criterion is identical for several splits and one split has to be selected at random. To obtain a deterministic behaviour during fitting, random_state has to be fixed to an integer.'}
# Parameter_8 =  {"Name":"max_leaf_nodes", "Type": ["int_or_null"], "Default_option":None, "Default_value":None, "Possible":["int"], 
#                "Definition":"Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes."}
# Parameter_9 =  {"Name":"min_impurity_decrease", "Type": ["float"], "Default_option":0.0, "Default_value":0.0, "Possible":["float"], 
#                "Definition":"A node will be split if this split induces a decrease of the impurity greater than or equal to this value.\n\nThe weighted impurity decrease equation is the following:\n\nN_t / N * (impurity - N_t_R / N_t * right_impurity - Nt_L / N_t * left_impurity)\n\nwhere N is the total number of samples, N_t is the number of samples at the current node, N_t_L is the number of samples in the left child, and N_t_R is the number of samples in the right child.\n\nN, N_t, N_t_R and N_t_L all refer to the weighted sum, if sample_weight is passed."}
# Parameter_10 =  {"Name":"class_weight", "Type": ["int_or_null"], "Default_option":None, "Default_value":None, "Possible":["int"], 
#                "Definition":"Weights associated with classes in the form {class_label: weight}. If None, all classes are supposed to have weight one. For multi-output problems, a list of dicts can be provided in the same order as the columns of y.\n\nNote that for multioutput (including multilabel) weights should be defined for each class of every column in its own dict. For example, for four-class multilabel classification weights should be [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of [{1:1}, {2:5}, {3:1}, {4:1}].\n\nThe “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))\n\nFor multi-output, the weights of each column of y will be multiplied.\n\nNote that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight is specified."}
# Parameter_11 =  {"Name":"ccp_alpha", "Type": ["float"], "Default_option":0.0, "Default_value":0.0, "Possible":["float"], 
#                "Definition":"Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen. By default, no pruning is performed."}

# Parameters = {"Parameter_0":Parameter_0, "Parameter_1":Parameter_1,"Parameter_2":Parameter_2,"Parameter_3":Parameter_3,
#              "Parameter_4":Parameter_4,"Parameter_5":Parameter_5,"Parameter_6":Parameter_6,
#              "Parameter_7":Parameter_7,"Parameter_8":Parameter_8,"Parameter_9":Parameter_9,
#              "Parameter_10":Parameter_10,"Parameter_11":Parameter_11}

# decision_tree_classifiier_algorithm = MLA(Name, Definition, Parameters)

# list_MLAs.append(decision_tree_classifiier_algorithm)


# # stochastic Gradient Decent
# Name = "StochasticGradientDecent"
# Definition = ["Linear classifiers (SVM, logistic regression, etc.) with SGD training.\nThis estimator implements regularized linear models with stochastic gradient descent (SGD) learning: the gradient of the loss is estimated each sample at a time and the model is updated along the way with a decreasing strength schedule (aka learning rate). SGD allows minibatch (online/out-of-core) learning via the partial_fit method. For best results using the default learning rate schedule, the data should have zero mean and unit variance.\nThis implementation works with data represented as dense or sparse arrays of floating point values for the features. The model it fits can be controlled with the loss parameter; by default, it fits a linear support vector machine (SVM).\nThe regularizer is a penalty added to the loss function that shrinks model parameters towards the zero vector using either the squared euclidean norm L2 or the absolute norm L1 or a combination of both (Elastic Net). If the parameter update crosses the 0.0 value because of the regularizer, the update is truncated to 0.0 to allow for learning sparse models and achieve online feature selection."]
# Parameter_0 = {"Name":"loss", "Type": ["option"], "Default_option":"hinge", "Default_value":"hinge", "Possible":['hinge', 'log_loss', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
#               "Definition":"The loss function to be used.\n\n‘hinge’ gives a linear SVM.\n\n‘log_loss’ gives logistic regression, a probabilistic classifier.\n\n‘modified_huber’ is another smooth loss that brings tolerance to\n\noutliers as well as probability estimates.\n\n‘squared_hinge’ is like hinge but is quadratically penalized.\n\n‘perceptron’ is the linear loss used by the perceptron algorithm.\n\nThe other losses, ‘squared_error’, ‘huber’, ‘epsilon_insensitive’ and ‘squared_epsilon_insensitive’ are designed for regression but can be useful in classification as well; see SGDRegressor for a description."}
# Parameter_1 = {"Name":"pentalty", "Type": ["option"], "Default_option":"l2", "Default_value":"l2", "Possible":['l2', 'l1', 'elasticnet', None],
#               "Definition":"The penalty (aka regularization term) to be used. Defaults to ‘l2’ which is the standard regularizer for linear SVM models. ‘l1’ and ‘elasticnet’ might bring sparsity to the model (feature selection) not achievable with ‘l2’. No penalty is added when set to None."}
# Parameter_2 = {"Name":"alpha", "Type": ["float"], "Default_option":0.0001, "Default_value":0.0001, "Possible":["float"],
#               "Definition":"Constant that multiplies the regularization term. The higher the value, the stronger the regularization. Also used to compute the learning rate when learning_rate is set to ‘optimal’. Values must be in the range [0.0, inf)."}
# Parameter_3 = {"Name":"l1_ratio", "Type": ["float"], "Default_option":0.15, "Default_value":0.15, "Possible":["float"],
#               "Definition":"The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1. l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1. Only used if penalty is ‘elasticnet’. Values must be in the range [0.0, 1.0]."}
# Parameter_4 = {"Name":"fit_intercept", "Type": ["bool"], "Default_option":True, "Default_value":True, "Possible":[True,False],
#               "Definition":"Whether the intercept should be estimated or not. If False, the data is assumed to be already centered."}
# Parameter_5 = {"Name":"max_iter", "Type": ["int"], "Default_option":1000, "Default_value":1000, "Possible":["int"],
#               "Definition":"The maximum number of passes over the training data (aka epochs). It only impacts the behavior in the fit method, and not the partial_fit method. Values must be in the range [1, inf)."}
# Parameter_6 = {"Name":"tol", "Type": ["float_or_null"], "Default_option":0.003, "Default_value":0.003, "Possible":["float"],
#               "Definition":"The stopping criterion. If it is not None, training will stop when (loss > best_loss - tol) for n_iter_no_change consecutive epochs. Convergence is checked against the training loss or the validation loss depending on the early_stopping parameter. Values must be in the range [0.0, inf)."}
# Parameter_7 = {"Name":"shuffle", "Type": ["bool"], "Default_option":True, "Default_value":True, "Possible":[True,False],
#               "Definition":"Whether or not the training data should be shuffled after each epoch."}
# Parameter_8 = {"Name":"verbose", "Type": ["int"], "Default_option":0, "Default_value":0, "Possible":["int"],
#               "Definition":"The verbosity level. Values must be in the range [0, inf)."}
# Parameter_9 = {"Name":"epsilon", "Type": ["float"], "Default_option":0.1, "Default_value":0.1, "Possible":["float"],
#               "Definition":"Epsilon in the epsilon-insensitive loss functions; only if loss is ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’. For ‘huber’, determines the threshold at which it becomes less important to get the prediction exactly right. For epsilon-insensitive, any differences between the current prediction and the correct label are ignored if they are less than this threshold. Values must be in the range [0.0, inf)."}
# Parameter_10 = {"Name":"n_jobs", "Type": ["int_or_null"], "Default_option":None, "Default_value":None, "Possible":["int"],
#               "Definition":"The number of CPUs to use to do the OVA (One Versus All, for multi-class problems) computation. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors"}
# Parameter_11 = {"Name":"random_state", "Type": ["int_or_null"], "Default_option":None, "Default_value":None, "Possible":["int"],
#               "Definition":"Used for shuffling the data, when shuffle is set to True. Pass an int for reproducible output across multiple function calls."}
# Parameter_12 = {"Name":"learning_rate", "Type": ["option"], "Default_option":"optimal", "Default_value":"optimal", "Possible":['adaptive', 'invscaling', 'optimal', 'constant'],
#               "Definition":"The learning rate schedule:\n\n‘constant’: eta = eta0\n\n‘optimal’: eta = 1.0 / (alpha * (t + t0)) where t0 is chosen by a heuristic proposed by Leon Bottou.\n\n‘invscaling’: eta = eta0 / pow(t, power_t)\n\n‘adaptive’: eta = eta0, as long as the training keeps decreasing. Each time n_iter_no_change consecutive epochs fail to decrease the training loss by tol or fail to increase validation score by tol if early_stopping is True, the current learning rate is divided by 5."}
# Parameter_13 = {"Name":"eta", "Type": ["float"], "Default_option":0.0, "Default_value":0.0, "Possible":["float"],
#               "Definition":"The initial learning rate for the ‘constant’, ‘invscaling’ or ‘adaptive’ schedules. The default value is 0.0 as eta0 is not used by the default schedule ‘optimal’. Values must be in the range (0.0, inf)"}
# Parameter_14 = {"Name":"power_t", "Type": ["float"], "Default_option":0.5, "Default_value":0.5, "Possible":["float"],
#               "Definition":"The exponent for inverse scaling learning rate [default 0.5]. Values must be in the range (-inf, inf)."}
# Parameter_15 = {"Name":"early_stopping", "Type": ["bool"], "Default_option":False, "Default_value":False, "Possible":[True,False],
#               "Definition":"Whether to use early stopping to terminate training when validation score is not improving. If set to True, it will automatically set aside a stratified fraction of training data as validation and terminate training when validation score returned by the score method is not improving by at least tol for n_iter_no_change consecutive epochs."}
# Parameter_16 = {"Name":"validation_fraction", "Type": ["float"], "Default_option":0.1, "Default_value":0.1, "Possible":["float"],
#               "Definition":"The proportion of training data to set aside as validation set for early stopping. Must be between 0 and 1. Only used if early_stopping is True. Values must be in the range (0.0, 1.0)."}
# Parameter_17 = {"Name":"n_iter_no_change", "Type": ["int"], "Default_option":5, "Default_value":5, "Possible":["int"],
#               "Definition":"Number of iterations with no improvement to wait before stopping fitting. Convergence is checked against the training loss or the validation loss depending on the early_stopping parameter. Integer values must be in the range [1, max_iter)."}
# Parameter_18 = {"Name":"class_weight", "Type": ["option"], "Default_option":None, "Default_value":None, "Possible":["balanced", None],
#               "Definition":"The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))."}
# Parameter_19 = {"Name":"warm_start", "Type": ["bool"], "Default_option":False, "Default_value":False, "Possible":[True,False],
#               "Definition":"When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution. \n\n Repeatedly calling fit or partial_fit when warm_start is True can result in a different solution than when calling fit a single time because of the way the data is shuffled. If a dynamic learning rate is used, the learning rate is adapted depending on the number of samples already seen. Calling fit resets this counter, while partial_fit will result in increasing the existing counter."}
# Parameter_20 = {"Name":"average", "Type": ["bool"], "Default_option":False, "Default_value":False, "Possible":[True,False],
#               "Definition":"When set to True, computes the averaged SGD weights across all updates and stores the result in the coef_ attribute. If set to an int greater than 1, averaging will begin once the total number of samples seen reaches average. So average=10 will begin averaging after seeing 10 samples. Integer values must be in the range [1, n_samples]."}

# Parameters = {"Parameter_0":Parameter_0, "Parameter_1":Parameter_1,"Parameter_2":Parameter_2,"Parameter_3":Parameter_3,
#              "Parameter_4":Parameter_4,"Parameter_5":Parameter_5,"Parameter_6":Parameter_6,
#              "Parameter_7":Parameter_7,"Parameter_8":Parameter_8,"Parameter_9":Parameter_9,
#              "Parameter_10":Parameter_10,"Parameter_11":Parameter_11, "Parameter_12":Parameter_12,
#              "Parameter_13":Parameter_13,"Parameter_14":Parameter_14, "Parameter_15":Parameter_15,
#              "Parameter_16":Parameter_16,"Parameter_17":Parameter_17, "Parameter_18":Parameter_18,
#              "Parameter_19":Parameter_19, "Parameter_20":Parameter_20}

# StochasticGradientDecent_algorithm = MLA(Name, Definition, Parameters)

# list_MLAs.append(StochasticGradientDecent_algorithm)


# # Gaussian Naiive Bayes
# Name = "GaussianNaiveBayes"
# Definition = ["Gaussian Naive Bayes (GaussianNB)."]
# Parameter_0 = {"Name":"var_smoothing", "Type": ["float"], "Default_option":0.000000001, "Default_value":0.000000001, "Possible":["float"],
#               "Definition":"Portion of the largest variance of all features that is added to variances for calculation stability."}

# Parameters = {"Parameter_0":Parameter_0}

# GaussianNaiveBayes_algorithm = MLA(Name, Definition, Parameters)

# list_MLAs.append(GaussianNaiveBayes_algorithm)

# # Random Forest Classifyer
# Name = "RandomForestClassifier"
# Definition = ["A random forest classifier.\n\nA random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is controlled with the max_samples parameter if bootstrap=True (default), otherwise the whole dataset is used to build each tree."]
# Parameter_0 = {"Name":"n_estimators", "Type": ["int"], "Default_option":100, "Default_value":100, "Possible":["int"],
#               "Definition":"The number of trees in the forest."}
# Parameter_1 = {"Name":"criterion", "Type": ["option"], "Default_option":"gini", "Default_value":"gini", "Possible":['gini', 'entropy', 'log_loss'],
#               "Definition":"The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “log_loss” and “entropy” both for the Shannon information gain, see Mathematical formulation."}
# Parameter_2 = {"Name":"max_depth", "Type": ["int_or_null"], "Default_option":None, "Default_value":None, "Possible":["int"],
#               "Definition":"The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples."}
# Parameter_3 = {"Name":"min_smaples_split", "Type": ["int"], "Default_option":2, "Default_value":2, "Possible":["int"],
#               "Definition":"The minimum number of samples required to split an internal node: consider min_samples_split as the minimum number."}
# Parameter_4 = {"Name":"min_samples_leaf", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
#               "Definition":"The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression. Consider min_samples_leaf as the minimum number."}
# Parameter_5 = {"Name":"min_weight_fraction_leaf", "Type": ["float"], "Default_option":0.0, "Default_value":0.0, "Possible":["float"],
#               "Definition":"The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided."}
# Parameter_6 = {"Name":"max_features", "Type": ["option"], "Default_option":"sqrt", "Default_value":"sqrt", "Possible":["sqrt","log2",None],
#               "Definition":"The number of features to consider when looking for the best split:\n\nIf None, then max_features=n_features.\n\nIf “log2”, then max_features=log2(n_features).\n\nIf “sqrt”, then max_features=sqrt(n_features)."}
# Parameter_7 = {"Name":"max_leaf_nodes", "Type": ["int_or_null"], "Default_option":None, "Default_value":None, "Possible":["int"],
#               "Definition":"Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes."}
# Parameter_8 = {"Name":"min_impurity_decrease", "Type": ["float"], "Default_option":0.0, "Default_value":0.0, "Possible":["float"],
#               "Definition":"A node will be split if this split induces a decrease of the impurity greater than or equal to this value.\n\nThe weighted impurity decrease equation is the following:\n\nN_t / N * (impurity - N_t_R / N_t * right_impurity - Nt_L / N_t * left_impurity)\n\nwhere N is the total number of samples, N_t is the number of samples at the current node, N_t_L is the number of samples in the left child, and N_t_R is the number of samples in the right child.\n\nN, N_t, N_t_R and N_t_L all refer to the weighted sum, if sample_weight is passed."}
# Parameter_9 = {"Name":"bootstrap", "Type": ["bool"], "Default_option":True, "Default_value":True, "Possible":[True,False],
#               "Definition":"Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree."}
# Parameter_10 = {"Name":"oob_score", "Type": ["bool"], "Default_option":False, "Default_value":False, "Possible":[True,False],
#               "Definition":"Whether to use out-of-bag samples to estimate the generalization score. Only available if bootstrap=True."}
# Parameter_11 = {"Name":"n_jobs", "Type": ["int_or_null"], "Default_option":None, "Default_value":None, "Possible":["int"],
#               "Definition":"The number of jobs to run in parallel. fit, predict, decision_path and apply are all parallelized over the trees. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors."}
# Parameter_12 = {"Name":"random_state", "Type": ["int_or_null"], "Default_option":None, "Default_value":None, "Possible":[""],
#               "Definition":"Controls both the randomness of the bootstrapping of the samples used when building trees (if bootstrap=True) and the sampling of the features to consider when looking for the best split at each node (if max_features < n_features)."}
# Parameter_13 = {"Name":"verbose", "Type": ["int"], "Default_option":0, "Default_value":0, "Possible":["int"],
#               "Definition":"Controls the verbosity when fitting and predicting."}
# Parameter_14 = {"Name":"warm_start", "Type": ["bool"], "Default_option":False, "Default_value":False, "Possible":[True,False],
#               "Definition":"When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest. See Glossary and Fitting additional weak-learners for details."}
# Parameter_15 = {"Name":"class_weight", "Type": ["option"], "Default_option":None, "Default_value":None, "Possible":["balanced","balanced_subsample",None],
#               "Definition":"Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one. For multi-output problems, a list of dicts can be provided in the same order as the columns of y."}
# Parameter_16 = {"Name":"ccp_alpha", "Type": ["float"], "Default_option":0.0, "Default_value":0.0, "Possible":["float"],
#               "Definition":"Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen. By default, no pruning is performed."}
# Parameter_17 = {"Name":"max_samples", "Type": ["int_or_null"], "Default_option":None, "Default_value":None, "Possible":["int"],
#               "Definition":"If bootstrap is True, the number of samples to draw from X to train each base estimator.\n\n If None (default), then draw X.shape[0] samples.\n\nIf int, then draw max_samples samples."}

# Parameters = {"Parameter_0":Parameter_0, "Parameter_1":Parameter_1,"Parameter_2":Parameter_2,"Parameter_3":Parameter_3,
#              "Parameter_4":Parameter_4,"Parameter_5":Parameter_5,"Parameter_6":Parameter_6,
#              "Parameter_7":Parameter_7,"Parameter_8":Parameter_8,"Parameter_9":Parameter_9,
#              "Parameter_10":Parameter_10,"Parameter_11":Parameter_11, "Parameter_12":Parameter_12,
#              "Parameter_13":Parameter_13,"Parameter_14":Parameter_14, "Parameter_15":Parameter_15,
#              "Parameter_16":Parameter_16,"Parameter_17":Parameter_17}

# RandomForestClassifier_algorithm = MLA(Name, Definition, Parameters)

# list_MLAs.append(RandomForestClassifier_algorithm)

# # Gaussian Process Classifier
# Name = "GaussianProcessClassifier"
# Definition = ["Gaussian process classification (GPC) based on Laplace approximation."]
# Parameter_0 = {"Name":"Optimizer", "Type": ["option"], "Default_option":"fmin_l_bfgs_b", "Default_value":"fmin_l_bfgs_b", "Possible":["fmin_l_bfgs_b","None"],
#                "Definition":"Per default, the ‘L-BFGS-B’ algorithm from scipy.optimize.minimize is used. If None is passed, the kernel’s parameters are kept fixed."}
# Parameter_1 = {"Name":"n_restarts_optimizer", "Type": ["int"], "Default_option":0, "Default_value":0, "Possible":["int"], 
#                 "Definition": "The number of restarts of the optimizer for finding the kernel’s parameters which maximize the log-marginal likelihood. The first run of the optimizer is performed from the kernel’s initial parameters, the remaining ones (if any) from thetas sampled log-uniform randomly from the space of allowed theta-values. If greater than 0, all bounds must be finite. Note that n_restarts_optimizer=0 implies that one run is performed."}
# Parameter_2 = {"Name":"max_iter_predict", "Type": ["int"], "Default_option":100, "Default_value":100, "Possible":["int"],
#               "Definition":"The maximum number of iterations in Newton’s method for approximating the posterior during predict. Smaller values will reduce computation time at the cost of worse results."}
# Parameter_3 = {"Name":"warm_start", "Type": ["bool"], "Default_option":False, "Default_value":False, "Possible":[True,False],
#               "Definition":"f warm-starts are enabled, the solution of the last Newton iteration on the Laplace approximation of the posterior mode is used as initialization for the next call of _posterior_mode(). This can speed up convergence when _posterior_mode is called several times on similar problems as in hyperparameter optimization."}
# Parameter_4 = {"Name":"copy_X_train", "Type": ["bool"], "Default_option":True, "Default_value":True, "Possible":[True,False],
#               "Definition":"If True, a persistent copy of the training data is stored in the object. Otherwise, just a reference to the training data is stored, which might cause predictions to change if the data is modified externally."}
# Parameter_5 = {"Name":"random_state", "Type": ["int_or_null"], "Default_option":None, "Default_value":None, "Possible":["int"],
#               "Definition":"Determines random number generation used to initialize the centers. Pass an int for reproducible results across multiple function calls."}
# Parameter_6 = {"Name":"multi_class", "Type": ["option"], "Default_option":"one_vs_rest", "Default_value":"one_vs_rest", "Possible":["one_vs_rest","one_vs_one"],
#               "Definition":"Specifies how multi-class classification problems are handled. Supported are ‘one_vs_rest’ and ‘one_vs_one’. In ‘one_vs_rest’, one binary Gaussian process classifier is fitted for each class, which is trained to separate this class from the rest. In ‘one_vs_one’, one binary Gaussian process classifier is fitted for each pair of classes, which is trained to separate these two classes. The predictions of these binary predictors are combined into multi-class predictions. Note that ‘one_vs_one’ does not support predicting probability estimates."}
# Parameter_7 = {"Name":"n_jobs", "Type": ["int_or_null"], "Default_option":None, "Default_value":None, "Possible":["int"],
#               "Definition":"The number of jobs to use for the computation: the specified multiclass problems are computed in parallel. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors."}


# Parameters = {"Parameter_0":Parameter_0, "Parameter_1":Parameter_1,"Parameter_2":Parameter_2,"Parameter_3":Parameter_3,
#              "Parameter_4":Parameter_4,"Parameter_5":Parameter_5,"Parameter_6":Parameter_6,
#              "Parameter_7":Parameter_7}

# GaussianProcessClassifyer_algorithm = MLA(Name, Definition, Parameters)

# list_MLAs.append(GaussianProcessClassifyer_algorithm)


# ## This is clustering algorithm not classification.
# # K-Means
# Name = "K-Means"
# Definition = ["K-Means clustering."]
# Parameter_0 =  {"Name":"n_clusters", "Type": ["int"], "Default_option":8, "Default_value":8, "Possible":["int"], 
#                "Definition":"The number of clusters to form as well as the number of centroids to generate."}
# Parameter_1 =  {
#     "Name":"initialization", 
#     "Type": ["option"], 
#     "Default_option":"k-means++", 
#     "Default_value":"k-means++", 
#     "Possible":["k-means++", "random"], 
#     "Definition":"Method for initialization:\n\n‘k-means++’ : selects initial cluster centroids using sampling based on an empirical probability distribution of the points’ contribution to the overall inertia. This technique speeds up convergence. The algorithm implemented is “greedy k-means++”. It differs from the vanilla k-means++ by making several trials at each sampling step and choosing the best centroid among them.\n\n‘random’: choose n_clusters observations (rows) at random from data for the initial centroids.\n\nIf an array is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.\n\nIf a callable is passed, it should take arguments X, n_clusters and a random state and return an initialization."}
# Parameter_2 =  {"Name":"n_int", "Type": ["int"], "Default_option":10, "Default_value":10, "Possible":["int"], 
#                "Definition":"Number of times the k-means algorithm is run with different centroid seeds. The final results is the best output of n_init consecutive runs in terms of inertia. Several runs are recommended for sparse high-dimensional problems (see Clustering sparse data with k-means).\n\nWhen n_init='auto', the number of runs depends on the value of init: 10 if using init='random', 1 if using init='k-means++'."}
# Parameter_3 =  {"Name":"max_iter", "Type": ["int"], "Default_option":300, "Default_value":300, "Possible":["int"], 
#                "Definition":"Maximum number of iterations of the k-means algorithm for a single run."}
# Parameter_4 =  {"Name":"tol", "Type": ["float"], "Default_option":0.0001, "Default_value":0.0001, "Possible":["float"], 
#                "Definition":"Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence."}
# Parameter_5 =  {"Name":"verbose", "Type": ["int"], "Default_option":0, "Default_value":0, "Possible":["int"], 
#                "Definition":"Verbosity mode."}
# Parameter_6 =  {"Name":"random_state", "Type": ["int_or_null"], "Default_option":None, "Default_value":None, "Possible":["int"], 
#                "Definition":"Determines random number generation for centroid initialization. Use an int to make the randomness deterministic."}
# Parameter_7 =  {"Name":"copy_x", "Type": ["bool"], "Default_option":True, "Default_value":True, "Possible":[True,False], 
#                "Definition":"When pre-computing distances it is more numerically accurate to center the data first. If copy_x is True (default), then the original data is not modified. If False, the original data is modified, and put back before the function returns, but small numerical differences may be introduced by subtracting and then adding the data mean. Note that if the original data is not C-contiguous, a copy will be made even if copy_x is False. If the original data is sparse, but not in CSR format, a copy will be made even if copy_x is False."}
# Parameter_8 =  {"Name":"algorithm", "Type": ["option"], "Default_option":"lloyd", "Default_value":"lloyd", "Possible":["lloyd", "elkan", "auto", "full"], 
#                "Definition":'K-means algorithm to use. The classical EM-style algorithm is "lloyd". The "elkan" variation can be more efficient on some datasets with well-defined clusters, by using the triangle inequality. However it’s more memory intensive due to the allocation of an extra array of shape (n_samples, n_clusters).\n\n"auto" and "full" are deprecated and they will be removed in Scikit-Learn 1.3. They are both aliases for "lloyd".'}

# Parameters = {"Parameter_0":Parameter_0, "Parameter_1":Parameter_1,"Parameter_2":Parameter_2,"Parameter_3":Parameter_3,
#              "Parameter_4":Parameter_4,"Parameter_5":Parameter_5,"Parameter_6":Parameter_6,
#              "Parameter_7":Parameter_7,"Parameter_8":Parameter_8}

# k_mean_algorithm = MLA(Name, Definition, Parameters)

# #list_MLAs.append(k_mean_algorithm)

# # Bernoulli Naive Bayes Classifier
# Name = "BernoulliNB"
# Definition = ["Naive Bayes classifier for multivariate Bernoulli models.\n\nLike MultinomialNB, this classifier is suitable for discrete data. The difference is that while MultinomialNB works with occurrence counts, BernoulliNB is designed for binary/boolean features."]
# Parameter_0 =  {"Name":"alpha", 
#                 "Type": ["float"], 
#                 "Default_option":1.0, 
#                 "Default_value":1.0, 
#                 "Possible":["float"], 
#                "Definition":"Additive (Laplace/Lidstone) smoothing parameter (set alpha=0 and force_alpha=True, for no smoothing)."}
# Parameter_1 =  {"Name":"force_alpha", 
#                 "Type": ["bool"], 
#                 "Default_option":False, 
#                 "Default_value":False, 
#                 "Possible":[True, False], 
#                "Definition":"If False and alpha is less than 1e-10, it will set alpha to 1e-10. If True, alpha will remain unchanged. This may cause numerical errors if alpha is too close to 0."}
# Parameter_2 =  {"Name":"binarize", 
#                 "Type": ["float_or_null"], 
#                 "Default_option":0.0, 
#                 "Default_value":0.0, 
#                 "Possible":["float"], 
#                "Definition":"Threshold for binarizing (mapping to booleans) of sample features. If None, input is presumed to already consist of binary vectors."}
# Parameter_3 =  {"Name":"fit_prior", 
#                 "Type": ["bool"], 
#                 "Default_option":True, 
#                 "Default_value":True, 
#                 "Possible":[True, False], 
#                "Definition":"Whether to learn class prior probabilities or not. If false, a uniform prior will be used."}
# # Did not add due to overcomplications to adding array impute values.
# #Parameter_4 =  {"Name":"class_prior", 
# #                "Type": ["bool"], 
# #                "Default_option":True, 
# #                "Default_value":True, 
# #                "Possible":[True, False], 
# #               "Definition":"Whether to learn class prior probabilities or not. If false, a uniform prior will be used."}


# Parameters = {"Parameter_0":Parameter_0, "Parameter_1":Parameter_1,"Parameter_2":Parameter_2,"Parameter_3":Parameter_3}

# BernoulliNB_algorithm = MLA(Name, Definition, Parameters)

# list_MLAs.append(BernoulliNB_algorithm)



# Name = "MultinomialNB"
# Definition = ["The multinomial Naive Bayes classifier \n is suitable for classification with discrete features (e.g., word counts for text classification). \n The multinomial distribution normally requires integer feature counts. However, in practice, fractional counts such as tf-idf may also work."]
# Parameter_0 =  {
#                 "Name":"alpha", 
#                 "Type": ["float"], 
#                 "Default_option":1.0, 
#                 "Default_value":1.0, 
#                 "Possible":["float"], 
#                "Definition":"Additive (Laplace/Lidstone) smoothing parameter (set alpha=0 and force_alpha=True, for no smoothing)."
#                }
# Parameter_1 =  {"Name":"force_alpha", 
#                 "Type": ["bool"], 
#                 "Default_option":False, 
#                 "Default_value":False, 
#                 "Possible":[True, False], 
#                "Definition":"If False and alpha is less than 1e-10, it will set alpha to 1e-10. If True, alpha will remain unchanged. This may cause numerical errors if alpha is too close to 0."}

# Parameter_2 =  {"Name":"fit_prior", 
#                 "Type": ["bool"], 
#                 "Default_option":True, 
#                 "Default_value":True, 
#                 "Possible":[True, False], 
#                "Definition":"Whether to learn class prior probabilities or not. If false, a uniform prior will be used."}

# Parameters = {"Parameter_0":Parameter_0, "Parameter_1":Parameter_1, "Parameter_2":Parameter_2 }

# MultinomialNB_algorithm = MLA(Name, Definition, Parameters)

# list_MLAs.append(MultinomialNB_algorithm)

# Name = "ComplementNB"
# Definition = ["The Complement Naive Bayes classifier was designed to correct the “severe assumptions” made by the standard Multinomial Naive Bayes classifier. \n It is particularly suited for imbalanced data sets."]
# Parameter_0 =  {
#                 "Name":"alpha", 
#                 "Type": ["float"], 
#                 "Default_option":1.0, 
#                 "Default_value":1.0, 
#                 "Possible":["float"], 
#                "Definition":"Additive (Laplace/Lidstone) smoothing parameter (set alpha=0 and force_alpha=True, for no smoothing)."
#                }
# Parameter_1 =  {"Name":"force_alpha", 
#                 "Type": ["bool"], 
#                 "Default_option":False, 
#                 "Default_value":False, 
#                 "Possible":[True, False], 
#                "Definition":"If False and alpha is less than 1e-10, it will set alpha to 1e-10. If True, alpha will remain unchanged. This may cause numerical errors if alpha is too close to 0."}

# Parameter_2 =  {"Name":"fit_prior", 
#                 "Type": ["bool"], 
#                 "Default_option":True, 
#                 "Default_value":True, 
#                 "Possible":[True, False], 
#                "Definition":"Only used in edge case with a single class in the training set."}

# Parameter_3 =  {"Name":"norm", 
#                 "Type": ["bool"], 
#                 "Default_option":False, 
#                 "Default_value":False, 
#                 "Possible":[True, False], 
#                "Definition":"Whether or not a second normalization of the weights is performed. The default behavior mirrors the implementations found in Mahout and Weka, which do not follow the full algorithm described in Table 9 of the paper."}




# Parameters = {"Parameter_0":Parameter_0, "Parameter_1":Parameter_1, "Parameter_2":Parameter_2, "Parameter_3" : Parameter_3 }

# ComplementNB_algorithm = MLA(Name, Definition, Parameters)

# list_MLAs.append(ComplementNB_algorithm)


# Name = "LogisticRegression"
# Definition = ["Logistic Regression is implemented as a linear model for classification rather than regression"]
# Parameter_0 =  {
#     "Name":"penalty", 
#     "Type": ["option"], 
#     "Default_option":"l2", 
#     "Default_value":"l2", 
#     "Possible":["l1", "l2", "elasticnet", "None"], 
#     "Definition":" Specify the norm of the penalty: \nNone: no penalty is added;\nl2: add a L2 penalty term and it is the default choice;\nl1: add a L1 penalty term;\nelasticnet: both L1 and L2 penalty terms are added."
#     }

# Parameter_1 =  {"Name":"dual", 
#                 "Type": ["bool"], 
#                 "Default_option":False, 
#                 "Default_value":False, 
#                 "Possible":[True, False], 
#                "Definition":"Dual formulation is only implemented for l2 penalty with liblinear solver. Prefer dual=False when n_samples > n_features."
#                }

# Parameter_2 =  {
#                 "Name":"tol", 
#                 "Type": ["float"], 
#                 "Default_option":0.0001, 
#                 "Default_value":0.0001, 
#                 "Possible":["float"], 
#                "Definition":"Tolerance for stopping criteria."
#                }

# Parameter_3 =  {"Name":"C", 
#                 "Type": ["float"], 
#                 "Default_option":1.0, 
#                 "Default_value":1.0, 
#                 "Possible":["float"], 
#                "Definition":"Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization."
#                }


# Parameter_4 =  {"Name":"fit_intercept", 
#                 "Type": ["bool"], 
#                 "Default_option":True, 
#                 "Default_value":True, 
#                 "Possible":[True, False], 
#                "Definition":"Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function."
#                }

# Parameter_5 =  {"Name":"intercept_scaling", 
#                 "Type": ["float"], 
#                 "Default_option":1.0, 
#                 "Default_value":1.0, 
#                 "Possible":["float"], 
#                "Definition":"Useful only when the solver ‘liblinear’ is used and self.fit_intercept is set to True. In this case, x becomes [x, self.intercept_scaling], i.e. a “synthetic” feature with constant value equal to intercept_scaling is appended to the instance vector. The intercept becomes intercept_scaling * synthetic_feature_weight."
#                }


# # solver{‘lbfgs’, ‘liblinear’, ‘newton-cg’, ‘newton-cholesky’, ‘sag’, ‘saga’}, default=’lbfgs’
# Parameter_6 =  {
#     "Name":"solver", 
#     "Type": ["option"], 
#     "Default_option":"lbfgs", 
#     "Default_value":"lbfgs", 
#     "Possible":["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"], 
#     "Definition":"Warning The choice of the algorithm depends on the penalty chosen. Supported penalties by solver:‘lbfgs’ - [‘l2’, None]‘liblinear’ - [‘l1’, ‘l2’]‘newton-cg’ - [‘l2’, None]‘newton-cholesky’ - [‘l2’, None]' sag’ - [‘l2’, None]‘saga’ - [‘elasticnet’, ‘l1’, ‘l2’, None]"
#     }

# Parameter_7 = {
#                "Name":"max_iter", 
#                "Type": ["int"], 
#                "Default_option":100,
#                "Default_value":100,
#                "Possible":["int"], 
#                "Definition":"Maximum number of iterations taken for the solvers to converge."
#                }

# Parameter_8 =  {
#     "Name":"multi_class", 
#     "Type": ["option"], 
#     "Default_option":"auto", 
#     "Default_value":"auto", 
#     "Possible":["auto", "multinomial", "ovr", "auto"], 
#     "Definition":"Specify the norm of the penalty: \nNone: no penalty is added;\nl2: add a L2 penalty term and it is the default choice;\nl1: add a L1 penalty term;\nelasticnet: both L1 and L2 penalty terms are added."
#     }


# Parameter_9 = {
#                "Name":"verbose", 
#                "Type": ["int"], 
#                "Default_option":0,
#                "Default_value":0,
#                "Possible":["int"], 
#                "Definition":"For the liblinear and lbfgs solvers set verbose to any positive number for verbosity."
#                }


# Parameter_10 =  {"Name":"warm_start", 
#                 "Type": ["bool"], 
#                 "Default_option":False, 
#                 "Default_value":False, 
#                 "Possible":[True, False], 
#                "Definition":"Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function."
#                }



# Parameters = {"Parameter_0":Parameter_0, "Parameter_1":Parameter_1, "Parameter_2":Parameter_2, "Parameter_3" : Parameter_3, "Parameter_4": Parameter_4, "Parameter_5": Parameter_5, "Parameter_6": Parameter_6,
#               "Parameter_7": Parameter_7, "Parameter_8": Parameter_8, "Parameter_9": Parameter_9, "Parameter_10" : Parameter_10 }

# LogisticRegressionNB_algorithm = MLA(Name, Definition, Parameters)

# list_MLAs.append(LogisticRegressionNB_algorithm)




# Name = "Perceptron"
# Definition = ["The Perceptron is another simple classification algorithm suitable for large scale learning"]
# Parameter_0 =  {
#     "Name":"penalty", 
#     "Type": ["option"], 
#     "Default_option":"None", 
#     "Default_value":"None", 
#     "Possible":["l1", "l2", "elasticnet", "None"], 
#     "Definition":" Specify the norm of the penalty: \nNone: no penalty is added;\nl2: add a L2 penalty term and it is the default choice;\nl1: add a L1 penalty term;\nelasticnet: both L1 and L2 penalty terms are added."
#     }


# Parameter_1 =  {
#                 "Name":"alpha", 
#                 "Type": ["float"], 
#                 "Default_option":0.0001, 
#                 "Default_value":0.0001, 
#                 "Possible":["float"], 
#                "Definition":"Constant that multiplies the regularization term if regularization is used."
#                }

# Parameter_2 =  {
#                 "Name":"l1_ratio", 
#                 "Type": ["float"], 
#                 "Default_option":0.15, 
#                 "Default_value":0.15, 
#                 "Possible":["float"], 
#                "Definition":"The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1. l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1. Only used if penalty='elasticnet'."
#                }


# Parameter_3 =  {"Name":"fit_intercept", 
#                 "Type": ["bool"], 
#                 "Default_option":True, 
#                 "Default_value":True, 
#                 "Possible":[True, False], 
#                "Definition":"Whether the intercept should be estimated or not. If False, the data is assumed to be already centered."
#                }



# Parameter_4 = {
#                "Name":"max_iter", 
#                "Type": ["int"], 
#                "Default_option":1000,
#                "Default_value":1000,
#                "Possible":["int"], 
#                "Definition":"The maximum number of passes over the training data (aka epochs)."
#                }


# Parameter_5 =  {
#                 "Name":"tol", 
#                 "Type": ["float"], 
#                 "Default_option":0.0001, 
#                 "Default_value":0.0001, 
#                 "Possible":["float"], 
#                "Definition":"Tolerance for stopping criteria."
#                }

# Parameter_6 =  {"Name":"shuffle", 
#                 "Type": ["bool"], 
#                 "Default_option":True, 
#                 "Default_value":True, 
#                 "Possible":[True, False], 
#                "Definition":"Whether or not the training data should be shuffled after each epoch."
#                }


# Parameter_7 = {
#                "Name":"verbose", 
#                "Type": ["int"], 
#                "Default_option":0,
#                "Default_value":0,
#                "Possible":["int"], 
#                "Definition":"The verbosity level."
#                }

# Parameter_8 =  {
#                 "Name":"eta0", 
#                 "Type": ["float"], 
#                 "Default_option":1, 
#                 "Default_value":1, 
#                 "Possible":["float"], 
#                "Definition":"Constant by which the updates are multiplied."
#                }


# Parameter_9 =  {"Name":"early_stopping", 
#                 "Type": ["bool"], 
#                 "Default_option":False, 
#                 "Default_value":True, 
#                 "Possible":[True, False], 
#                "Definition":"Whether to use early stopping to terminate training when validation. score is not improving. If set to True, it will automatically set aside a stratified fraction of training data as validation and terminate training when validation score is not improving by at least tol for n_iter_no_change consecutive epochs."
#                }

# Parameter_10 =  {
#                 "Name":"validation_fraction", 
#                 "Type": ["float"], 
#                 "Default_option":0.1, 
#                 "Default_value":0.1, 
#                 "Possible":["float"], 
#                "Definition":"The proportion of training data to set aside as validation set for early stopping. Must be between 0 and 1. Only used if early_stopping is True."
#                }

# Parameter_11 = {
#                "Name":"n_iter_no_change", 
#                "Type": ["int"], 
#                "Default_option":5,
#                "Default_value":5,
#                "Possible":["int"], 
#                "Definition":"Number of iterations with no improvement to wait before early stopping."
#                }

# Parameter_12 =  {"Name":"warm_start", 
#                 "Type": ["bool"], 
#                 "Default_option":False, 
#                 "Default_value":True, 
#                 "Possible":[True, False], 
#                "Definition":"When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution."
#                }
# Parameter_13 =  {"Name":"random_state",
#                 "Type": ["int_or_null"], 
#                 "Default_option":0, 
#                 "Default_value":0,
#                 "Possible":["int"], 
#                 "Definition":"Used to shuffle the training data, when shuffle is set to True. Pass an int for reproducible output across multiple function calls."}

# Parameters = {"Parameter_0":Parameter_0, "Parameter_1":Parameter_1, "Parameter_2":Parameter_2, "Parameter_3" : Parameter_3, "Parameter_4": Parameter_4, "Parameter_5": Parameter_5, "Parameter_6": Parameter_6,
#               "Parameter_7": Parameter_7, "Parameter_8": Parameter_8, "Parameter_9": Parameter_9, "Parameter_10" : Parameter_10, "Parameter_11" : Parameter_11, "Parameter_12" : Parameter_12, "Parameter_13" : Parameter_13 }

# Perceptron_algorithm = MLA(Name, Definition, Parameters)

# list_MLAs.append(Perceptron_algorithm)


def getEnsembles():
    return list_Ensembles



# Method to pull all parameters from a given method
def getParameters(algorithm_Name):
    Parameters = {}
    for algorithm in list_Ensembles:
        if algorithm.getName() == algorithm_Name:
            Parameters = algorithm.getParameters()
    return Parameters

# get settings to fill the model parameters.
def getSettings(data, Parameters):
    settings = {}
    settings["Parameter_Length"] = len(Parameters)

    # Cycle through parameters
    for i in range(len(Parameters)):
        temp = Parameters["Parameter_" + str(i)]["Name"]+"_checked"

        # get name
        settings["Parameter_" + str(i) + "_Name"] = Parameters["Parameter_" + str(i)]["Name"]

        temp = Parameters["Parameter_" + str(i)]["Type"]
        if Parameters["Parameter_" + str(i)]["Type"][0] == "option_float":
            settings["Parameter_" + str(i)] = data[Parameters["Parameter_" + str(i)]["Name"] + "_Input"]
            settings["Parameter_" + str(i)] = convertToType(settings["Parameter_" + str(i)], Parameters["Parameter_" + str(i)]["Type"], data[Parameters["Parameter_" + str(i)]["Name"] + "_Float_Input"])
        else:
            settings["Parameter_" + str(i)] = data[Parameters["Parameter_" + str(i)]["Name"] + "_Input"]
            settings["Parameter_" + str(i)] = convertToType(settings["Parameter_" + str(i)], Parameters["Parameter_" + str(i)]["Type"])

    return settings

# Convert into different data types
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



# Method to create the model based on given form data.
def createModel(data):
    Parameters = getParameters(data["EnsembleAlgorithm"])
    settings = getSettings(data, Parameters)
    # xgb_classifier = xgb.XGBClassifier()
    # Check which alorithm to create
    if data["EnsembleAlgorithm"] == "XGBClassifier": 
       # Fill the model information, note that the parameters are in the same order as writen above.
        model = XGBClassifier(n_estimators=settings['Parameter_0'],
                              max_depth=settings['Parameter_1'],
                              max_leaves=settings['Parameter_2'],
                              learning_rate=settings["Parameter_3"],
                              objective=settings['Parameter_4'],
                              booster=settings['Parameter_5']
                                     )
    elif data["EnsembleAlgorithm"] == "LGBMClassifier": 
       # Fill the model information, note that the parameters are in the same order as writen above.
        model = LGBMClassifier(boosting_type=settings['Parameter_0'],
                              num_leaves=settings['Parameter_1'],
                              learning_rate=settings['Parameter_2'],
                              n_estimators=settings["Parameter_3"],
                              objective=settings['Parameter_4'],
                              reg_alpha=settings['Parameter_5'],
                              reg_lambda=settings['Parameter_6']
                                     )
        
    elif data["EnsembleAlgorithm"] == "CatBoost":
        model = CatBoostClassifier(loss_function=settings['Parameter_0'],
                                   iterations=settings["Parameter_1"],
                                   learning_rate=settings["Parameter_2"],
                                   depth = settings["Parameter_3"],
                                   l2_leaf_reg= settings["Parameter_4"], 
                                   random_strength = settings["Parameter_5"],
                                   border_count = settings["Parameter_6"],
                                   bagging_temperature=settings["Parameter_7"]
                                   )
    

    # elif data["MLalgorithm"] == "SVM": 
    #     model = SVC(C=settings['Parameter_0'],
    #                 kernel=settings['Parameter_1'],
    #                 degree=settings['Parameter_2'],
    #                 gamma=settings['Parameter_3'],
    #                 coef0=settings['Parameter_4'],
    #                 shrinking=settings['Parameter_5'],
    #                 probability=settings['Parameter_6'],
    #                 tol=settings['Parameter_7'],
    #                 cache_size=settings['Parameter_8'],
    #                 class_weight=settings['Parameter_9'],
    #                 verbose=settings['Parameter_10'],
    #                 max_iter=settings['Parameter_11'],
    #                 decision_function_shape=settings['Parameter_12'],
    #                 break_ties=settings['Parameter_13'],
    #                 random_state=settings['Parameter_14'])

    # elif data["MLalgorithm"] == "K-Means":
    #     model = KMeans(n_clusters=settings['Parameter_0'],
    #                 init=settings['Parameter_1'],
    #                 n_init=settings['Parameter_2'],
    #                 max_iter=settings['Parameter_3'],
    #                 tol=settings['Parameter_4'],
    #                 verbose=settings['Parameter_5'],
    #                 random_state=settings['Parameter_6'],
    #                 copy_x=settings['Parameter_7'],
    #                 algorithm=settings['Parameter_8'])

    # elif data["MLalgorithm"] == "DecisionTreeClassifier":
    #     model = DecisionTreeClassifier(criterion=settings['Parameter_0'],
    #                 splitter=settings['Parameter_1'],
    #                 max_depth=settings['Parameter_2'],
    #                 min_samples_split=settings['Parameter_3'],
    #                 min_samples_leaf=settings['Parameter_4'],
    #                 min_weight_fraction_leaf=settings['Parameter_5'],
    #                 max_features=settings['Parameter_6'],
    #                 random_state=settings['Parameter_7'],
    #                 max_leaf_nodes=settings['Parameter_8'],
    #                 min_impurity_decrease=settings['Parameter_9'],
    #                 class_weight=settings['Parameter_10'],
    #                 ccp_alpha=settings['Parameter_11'])

    # elif data["MLalgorithm"] == "GaussianNaiveBayes":
    #     model = GaussianNB(var_smoothing=settings['Parameter_0'])


    # elif data["MLalgorithm"] == "StochasticGradientDecent":
    #     model = SGDClassifier(loss=settings['Parameter_0'],
    #                 penalty=settings['Parameter_1'],
    #                 alpha=settings['Parameter_2'],
    #                 l1_ratio=settings['Parameter_3'],
    #                 fit_intercept=settings['Parameter_4'],
    #                 max_iter=settings['Parameter_5'],
    #                 tol=settings['Parameter_6'],
    #                 shuffle=settings['Parameter_7'],
    #                 verbose=settings['Parameter_8'],
    #                 epsilon=settings['Parameter_9'],
    #                 n_jobs=settings['Parameter_10'],
    #                 random_state=settings['Parameter_11'],
    #                 learning_rate=settings['Parameter_12'],
    #                 eta0=settings['Parameter_13'],
    #                 power_t=settings['Parameter_14'],
    #                 early_stopping=settings['Parameter_15'],
    #                 validation_fraction=settings['Parameter_16'],
    #                 n_iter_no_change=settings['Parameter_17'],
    #                 class_weight=settings['Parameter_18'],
    #                 warm_start=settings['Parameter_19'],
    #                 average=settings['Parameter_20'])

    # elif data["MLalgorithm"] == "RandomForestClassifier":
    #     model = RandomForestClassifier(n_estimators=settings['Parameter_0'],
    #                 criterion=settings['Parameter_1'],
    #                 max_depth=settings['Parameter_2'],
    #                 min_samples_split=settings['Parameter_3'],
    #                 min_samples_leaf=settings['Parameter_4'],
    #                 min_weight_fraction_leaf=settings['Parameter_5'],
    #                 max_features=settings['Parameter_6'],
    #                 max_leaf_nodes=settings['Parameter_7'],
    #                 min_impurity_decrease=settings['Parameter_8'],
    #                 bootstrap=settings['Parameter_9'],
    #                 oob_score=settings['Parameter_10'],
    #                 n_jobs=settings['Parameter_11'],
    #                 random_state=settings['Parameter_12'],
    #                 verbose=settings['Parameter_13'],
    #                 warm_start=settings['Parameter_14'],
    #                 class_weight=settings['Parameter_15'],
    #                 ccp_alpha=settings['Parameter_16'],
    #                 max_samples=settings['Parameter_17'])



    # elif data["MLalgorithm"] == "GaussianProcessClassifier":
    #     model = GaussianProcessClassifier(optimizer=settings['Parameter_0'],
    #                 n_restarts_optimizer=settings['Parameter_1'],
    #                 max_iter_predict=settings['Parameter_2'],
    #                 warm_start=settings['Parameter_3'],
    #                 copy_X_train=settings['Parameter_4'],
    #                 random_state=settings['Parameter_5'],
    #                 multi_class=settings['Parameter_6'],
    #                 n_jobs=settings['Parameter_7'])
        
    # elif data["MLalgorithm"] == "BernoulliNB":
    #     model = BernoulliNB(alpha=settings['Parameter_0'],
    #                         force_alpha=settings['Parameter_1'],
    #                         binarize=settings['Parameter_2'],
    #                         fit_prior=settings['Parameter_3'])
    
    # elif data["MLalgorithm"] == "MultinomialNB":
    #     model = MultinomialNB(alpha=settings['Parameter_0'],
    #                         force_alpha=settings['Parameter_1'],
    #                         fit_prior=settings['Parameter_2'])
    
    # elif data["MLalgorithm"] == "ComplementNB":
    #     model = ComplementNB(alpha = settings['Parameter_0'],
    #                         force_alpha = settings['Parameter_1'],
    #                         fit_prior = settings['Parameter_2'],
    #                         norm = settings['Parameter_3'])
    
    # elif data["MLalgorithm"] == "LogisticRegression":
    #     model = LogisticRegression(penalty = settings['Parameter_0'],
    #                                dual = settings["Parameter_1"],
    #                                tol = settings["Parameter_2"],
    #                                C = settings["Parameter_3"],
    #                                fit_intercept = settings["Parameter_4"],
    #                                intercept_scaling = settings["Parameter_5"],
    #                                solver = settings["Parameter_6"],
    #                                max_iter = settings["Parameter_7"],
    #                                multi_class = settings["Parameter_8"],
    #                                verbose = settings["Parameter_9"],
    #                                warm_start = settings["Parameter_10"])
    
    # elif data["MLalgorithm"] == "Perceptron":
    #     model = Perceptron(penalty = settings['Parameter_0'],
    #                                alpha = settings["Parameter_1"],
    #                                l1_ratio = settings["Parameter_2"],
    #                                fit_intercept = settings["Parameter_3"],
    #                                max_iter = settings["Parameter_4"],
    #                                tol = settings["Parameter_5"],
    #                                shuffle = settings["Parameter_6"],
    #                                verbose = settings["Parameter_7"],
    #                                eta0 = settings["Parameter_8"],
    #                                early_stopping = settings["Parameter_9"],
    #                                validation_fraction = settings["Parameter_10"],
    #                                n_iter_no_change = settings["Parameter_11"],
    #                                warm_start = settings["Parameter_12"])
          
        
    return model, settings