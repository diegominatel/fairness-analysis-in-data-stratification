import math
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from aif360.sklearn.inprocessing import AdversarialDebiasing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

def set_configs(n_columns):
    
    adversarial_debiasing = {
        'AD' : [AdversarialDebiasing,
                {'prot_attr' : ['Group'],
                 'num_epochs' : list(range(50, 500, 30)),
                 'random_state' : [12345]}]
    }
    
    decision_tree = {
        'DT' : [DecisionTreeClassifier,
                {'criterion' : ['gini'],
                 'min_samples_leaf' : [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
                 'min_samples_split' : [5], 
                 'random_state' : [12345]}]
    }
    
    mlp = {
        'MLP' : [MLPClassifier,
                 {'hidden_layer_sizes' : list(range(5, 20, 1)),
                  'random_state' : [12345]}]
    }
    
    random_forest = {
        'RF' : [RandomForestClassifier,
                {'n_estimators' : list(range(100, 475, 25)),
                 'min_samples_split' : [math.floor(abs(math.sqrt(n_columns - 1)))],
                 'n_jobs' : [4],
                 'random_state' : [12345]}]
    }
    
    svm = {
        'SVM' : [SVC,
                 {'kernel' : ['rbf'], 'C' : [1], 'gamma' : list(np.arange(0.0025, 1.075, 0.075)), 
                  'random_state' : [12345]}]
    }
    
    xgb = {
        'XGB' : [XGBClassifier,
                 {'n_estimators' : list(range(100, 475, 25)),
                  'random_state' : [12345]}]
    }
    
    
    all_configs = {
        'ad'     : adversarial_debiasing,
        'dt'     : decision_tree,
        'mlp'    : mlp,
        'rf'     : random_forest,
        'svm'    : svm,
        'xgb'    : xgb
    }
    
    return all_configs