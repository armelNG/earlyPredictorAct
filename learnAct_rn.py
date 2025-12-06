# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 22:40:54 2024

@author: NGATCH04
"""
import time
import matplotlib.pylab
import numpy as np
import pandas as pdact
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import randint
from sklearn import metrics
import pandas as pdact
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve

from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as pyplot

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

Md1Num_act = pdact.read_csv('./results/matrices/act/Md1Num_act.csv', sep=',', dtype=object)
Md1Num_act = Md1Num_act.set_index('code_etudiant')
print(Md1Num_act)

# Separation du data set - Set d'apprentissage
Md1app_act = Md1Num_act[Md1Num_act.type == 'app']
y_app = Md1app_act['result']
x_app = Md1app_act.drop('result', axis=1).drop('type', axis=1)

# Separation du data set - Set de test
Md1test_act = Md1Num_act[Md1Num_act.type == 'test']
y_test = Md1test_act['result']
x_test = Md1test_act.drop('result', axis=1).drop('type', axis=1)

# Numérisation des y_test
yNum_test = y_test.replace('Pass',0)
yNum_test = yNum_test.replace('Fail',1)
yNum_test = yNum_test.replace('Dropout',2)
print(yNum_test)

# Numérisation des y_app
yNum_app = y_app.replace('Pass',0)
yNum_app = yNum_app.replace('Fail',1)
yNum_app = yNum_app.replace('Dropout',2)
print(yNum_app)

"""
# Instantiation du Random Forest
# rf = RandomForestClassifier(n_estimators = 5000, random_state = 3)
rf = RandomForestClassifier(
     n_estimators=3822,
     criterion='gini',
     max_depth=15,
     min_samples_split=2,
     min_samples_leaf=1,
     min_weight_fraction_leaf=0.0,
     max_features='sqrt',
     max_leaf_nodes=None,
     min_impurity_decrease=0.0,
     bootstrap=True,
     oob_score=False,
     n_jobs=None,
     random_state=3,
     verbose=0,
     warm_start=False,
     class_weight=None,
     ccp_alpha=0.0,
     max_samples=None,)
"""

#----------------------------------------------------------------------------------
# Evaluate run time and prediction accuracy
def evaluate_model(model, x_train, y_train, x_test, y_test):
    n_trees = model.get_params()['n_estimators']
    n_features = x_train.shape[1]
    
    # Train and predict 10 times to evaluate time and accuracy
    predictions = []
    run_times = []
    for _ in range(10):
        start_time = time.time()
        model.fit(x_train, y_train)
        predictions.append(model.predict(x_test))
    
        end_time = time.time()
        run_times.append(end_time - start_time)
    
    # Run time and predictions need to be averaged
    run_time = np.mean(run_times)
    predictions = np.mean(np.array(predictions), axis = 0)
    
    # Calculate performance metrics
    errors = abs(predictions - y_test)
    mean_error = np.mean(errors)
    mape = 100 * np.mean(errors / y_test)
    accuracy = 100 - mape
    
    # Return results in a dictionary
    results = {'time': run_time, 'error': mean_error, 'accuracy': accuracy, 'n_trees': n_trees, 'n_features': n_features}
    
    return results

# Plot results comparaison score vs NumTrees et Training Time vs Num Trees
def plot_results(model, param = 'n_estimators', name = 'Num Trees'):
    param_name = 'param_%s' % param

    # Extract information from the cross validation model
    train_scores = model.cv_results_['mean_train_score']
    test_scores = model.cv_results_['mean_test_score']
    train_time = model.cv_results_['mean_fit_time']
    param_values = list(model.cv_results_[param_name])
    
    # Plot the scores over the parameter
    plt.subplots(1, 2, figsize=(10, 6))
    plt.subplot(121)
    plt.plot(param_values, train_scores, 'bo-', label = 'train')
    plt.plot(param_values, test_scores, 'go-', label = 'test')
    plt.ylim(ymin = -10, ymax = 0)
    plt.legend()
    plt.xlabel(name)
    plt.ylabel('Neg Mean Absolute Error')
    plt.title('Score vs %s' % name)
    
    plt.subplot(122)
    plt.plot(param_values, train_time, 'ro-')
    plt.ylim(ymin = 0.0, ymax = 2.0)
    plt.xlabel(name)
    plt.ylabel('Train Time (sec)')
    plt.title('Training Time vs %s' % name)
    
    
    plt.tight_layout(pad = 4)
    
    
# Plot courbe d'apprentissage
def plot_learning_curves(model, X1, y1):
    mlp=MLPClassifier(activation="relu",max_iter=1500, hidden_layer_sizes=13, learning_rate='adaptive')
    X_train, X_val, y_train, y_val = train_test_split(X1, y1, test_size=0.2)
    
    mlp.fit(X_train,y_train)
    print (mlp.score(X_train,y_train))
    plt.plot(mlp.loss_curve_)
    mlp.fit(X_val,y_val)
    plt.plot(mlp.loss_curve_)
    plt.ylabel('Cost', fontsize = 14)
    plt.xlabel('Iterations', fontsize = 14)
    plt.title('Loss Curve', fontsize = 18, y = 1.03)
    plt.legend()
    plt.ylim(0,1.4)
    
#--------------------------------------------------------------------------------------------

# RandomSearch Grid
#============================================================================================
activation = ['identity','logistic','tanh', 'relu']
solver = ['lbfgs','sgd','adam']
learning_rate = ['adaptive','invscalling','constant']
alpha = np.logspace(-5, 0, 10)
max_iter = [1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000]
hidden_layer_sizes = np.arange(10, 15)
random_state = [0,1,2,3,4,5,6,7,8,9]
random_grid = {'activation': activation,
               'solver': solver,
               'learning_rate': learning_rate,
               'alpha': alpha,
               'max_iter': max_iter,
               'hidden_layer_sizes': hidden_layer_sizes,
               'random_state': random_state
               }

rn = MLPClassifier()
rand_search = RandomizedSearchCV(rn, 
                                 param_distributions = random_grid, 
                                 cv=3, 
                                 verbose=0
                                 )
#============================================================================================

rand_search.fit(x_app, yNum_app)
print(rand_search.cv_results_)
best_rn = rand_search.best_estimator_
print('Best hyperparameters:',  rand_search.best_params_)


#grid_final_accuracy = evaluate_model(best_rf, x_app, yNum_app, x_test, yNum_test)
#print(grid_final_accuracy)

best_rn.fit(x_app, yNum_app)
predictions = best_rn.predict(x_test)
dfpredictions = pdact.DataFrame(predictions, columns=['Results'])


# Matrices de confusion
# Create the confusion matrix
cm = metrics.confusion_matrix(yNum_test, predictions)
metrics.ConfusionMatrixDisplay(confusion_matrix=cm).plot();
plt.show()

'''
# Importance des variables
from sklearn.linear_model import Perceptron
clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(x_app, yNum_app)
coeffs = clf.coef_
print(pdact.DataFrame(coeffs, index = x_app.columns, columns = ["importance"]).sort_values("importance", ascending = False))
plt.figure(figsize=(15,10))
plt.bar([i for i in range(len(coeffs))], coeffs, tick_label=x_app.columns)
plt.show()
'''
# Plot results of algorithm
# plot_results(rand_search)

# Plot courbe d'apprentissage
# plot_learning_curves(best_rn, x_app, yNum_app)


#------------------------------------------------------------------------------------------------

# Generation du fichier des résultats

R = pdact.DataFrame()
for i in range(1, 30):
    best_rn.fit(x_app.iloc[:,0:i], y_app)
    result = best_rn.predict(x_test.iloc[:,0:i])
    print(result)
    R[i] = result
print(R)
R.to_csv('./results/matrices/act/ResultAct_rn.csv', index=True)

#------------------------------------------------------------------------------------------------

# Generation des matrices de confusion

for i in range(1, 30):
    best_rn.fit(x_app.iloc[:,0:i], yNum_app)
    result = best_rn.predict(x_test.iloc[:,0:i])
    cm1 = metrics.confusion_matrix(yNum_test, result)
    metrics.ConfusionMatrixDisplay(confusion_matrix=cm1).plot();
    plt.show()



# Classification report -------------------------------------------------------------------------
report = classification_report(yNum_test, predictions)
print(report)


# Construction ROC AUC Curve --------------------------------------------------------------------
pred_prob = best_rn.predict_proba(x_test)
y_test_binarized=label_binarize(yNum_test, classes =np.unique(yNum_test))
classes =np.unique(yNum_test)
# roc curve for classes
fpr = {}
tpr = {}
thresh ={}
roc_auc = dict()
n_class = classes.shape[0]
for i in range(n_class):    
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test_binarized[:,i], pred_prob[:,i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    # plotting    
    pyplot.plot(fpr[i], tpr[i], linestyle='-', 
             label='%s vs Rest (AUC=%0.2f)'%(classes[i],roc_auc[i]))
pyplot.plot([0,1],[0,1],'b--')
pyplot.xlim([0,1])
pyplot.ylim([0,1.05])
fpr, tpr, thresholds = roc_curve(y_test_binarized[:,i], pred_prob[:,i])
pyplot.title('Multiclass ROC Curve')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend(loc='lower right')
plt.show()

'''
# Performances du Système ----------------------------------------------------------------------------------------------------------------------------
predictionsSys = [2,2,0,0,2,1,2,0,0,2,0,0,0,2,0,2,2,0,2,0,2,0,2,2,0,0,1,2,0,2,2,0,2,2,0,0,2,0,0,2,0,2,0,0,0,2,2,0,2,0,2,2,0,2,2,0,0,0,2,0,1,0,2,2,0,0,0,2,
                  2,2,0,2,0,2,2,0,0,2,0,0,2,0,0,2,0,2,0,2,0,0,0,2,2,0,2,2,2,0,2,0,2,0,2,0,1,2,2,0,0,2,2,0,0,1,0,0,2,2,0,0,2,0,2,2,0,2,2,0,0,2,0,2,2,0,2,0,
                  0,2,0,2,0,2,0,2,0,2,0,0,2,0,2,0,0,2,0,0,2,0,2,1,0,1,0,2,0,0,0,2,0,2,0,2,0,0,2,2,1,0,0,0,2,0,0,0,2,2,0,0,2,0,0,0,2,0,0,2,0,2,0,2]
print(predictionsSys)
cmSys = metrics.confusion_matrix(yNum_test, predictionsSys)
metrics.ConfusionMatrixDisplay(confusion_matrix=cmSys).plot();
plt.show()

pred_prodSys = pdact.DataFrame()

pred_prodSys = np.array([[0,0.0344827586206897,0.96551724137931],[0,0,1],[1,0,0],[1,0,0],[0,0.0689655172413793,0.931034482758621],
                         [0.482758620689655,0.0689655172413793,0.448275862068966],[0,0,1],[1,0,0],[1,0,0],[0.172413793103448,0,0.827586206896552],[1,0,0],
                         [0.896551724137931,0,0.103448275862069],[1,0,0],[0,0,1],[1,0,0],[0.758620689655172,0,0.241379310344828],[0,0,1],
                         [0.827586206896552,0.0344827586206897,0.137931034482759],[0,0,1],[1,0,0],[0,0,1],[1,0,0],[0,0,1],[0,0,1],[1,0,0],[1,0,0],
                         [0.137931034482759,0.586206896551724,0.275862068965517],[0,0,1],[1,0,0],[0,0,1],[0.206896551724138,0.448275862068966,0.344827586206897],[1,0,0],
                         [0,0,1],[0,0,1],[1,0,0],[1,0,0],[0,0,1],[0.586206896551724,0.137931034482759,0.275862068965517],[1,0,0],[0,0.172413793103448,0.827586206896552],
                         [1,0,0],[0,0,1],[1,0,0],[1,0,0],[1,0,0],[0.241379310344828,0.482758620689655,0.275862068965517],[0,0.310344827586207,0.689655172413793],
                         [0.896551724137931,0,0.103448275862069],[0,0,1],[1,0,0],[0,0,1],[0,0,1],[1,0,0],[0,0,1],[0,0,1],[1,0,0],[1,0,0],
                         [0.96551724137931,0.0344827586206897,0],[0,0,1],[1,0,0],[0.137931034482759,0.586206896551724,0.275862068965517],
                         [0.724137931034483,0,0.275862068965517],[0,0,1],[0,0,1],[1,0,0],[1,0,0],[1,0,0],[0,0,1],[0,0,1],[0,0.379310344827586,0.620689655172414],[1,0,0],
                         [0,0,1],[1,0,0],[0,0,1],[0,0,1],[1,0,0],[1,0,0],[0,0,1],[1,0,0],[1,0,0],[0,0,1],[1,0,0],[1,0,0],[0,0,1],[1,0,0],[0,0,1],[1,0,0],[0,0,1],[1,0,0],
                         [1,0,0],[1,0,0],[0,0.0344827586206897,0.96551724137931],[0,0,1],[1,0,0],[0,0,1],[0.206896551724138,0,0.793103448275862],[0,0,1],[1,0,0],
                         [0,0.103448275862069,0.896551724137931],[1,0,0],[0,0,1],[1,0,0],[0,0,1],[1,0,0],[0.0344827586206897,0.310344827586207,0.655172413793103],
                         [0,0.758620689655172,0.241379310344828],[0,0,1],[1,0,0],[1,0,0],[0,0,1],[0,0,1],[1,0,0],[1,0,0],[0.448275862068966,0.551724137931034,0],[1,0,0],
                         [1,0,0],[0,0,1],[0,0,1],[1,0,0],[1,0,0],[0,0,1],[1,0,0],[0,0,1],[0.0689655172413793,0.0344827586206897,0.896551724137931],[1,0,0],
                         [0.586206896551724,0,0.413793103448276],[0,0,1],[1,0,0],[1,0,0],[0,0,1],[1,0,0],[0,0,1],[0,0,1],[1,0,0],[0.0689655172413793,0,0.931034482758621],
                         [1,0,0],[1,0,0],[0,0.0689655172413793,0.931034482758621],[1,0,0],[0,0,1],[1,0,0],[0,0,1],[1,0,0],[0,0,1],[1,0,0],
                         [0,0.0344827586206897,0.96551724137931],[1,0,0],[1,0,0],[0,0,1],[1,0,0],[0,0,1],[1,0,0],[1,0,0],[0,0,1],[1,0,0],
                         [0.931034482758621,0,0.0689655172413793],[0,0,1],[1,0,0],[0,0,1],[0.0344827586206897,0.482758620689655,0.482758620689655],[1,0,0],
                         [0,0.793103448275862,0.206896551724138],[1,0,0],[0,0,1],[1,0,0],[1,0,0],[0.724137931034483,0.0689655172413793,0.206896551724138],[0,0,1],
                         [1,0,0],[0,0,1],[0.724137931034483,0,0.275862068965517],[0,0.0689655172413793,0.931034482758621],[1,0,0],[1,0,0],[0,0,1],
                         [0,0.0689655172413793,0.931034482758621],[0,0.0689655172413793,0.931034482758621],[1,0,0],[1,0,0],[1,0,0],[0,0,1],[1,0,0],[1,0,0],[1,0,0],
                         [0,0.206896551724138,0.793103448275862],[0.0689655172413793,0,0.931034482758621],[1,0,0],[0.724137931034483,0,0.275862068965517],
                         [0.620689655172414,0,0.379310344827586],[1,0,0],[1,0,0],[1,0,0],[0,0,1],[1,0,0],[1,0,0],[0.413793103448276,0,0.586206896551724],[1,0,0],
                         [0,0.103448275862069,0.896551724137931],[1,0,0],[0.0344827586206897,0,0.96551724137931]])

print(pred_prodSys)
y_test_binarized=label_binarize(yNum_test, classes =np.unique(yNum_test))
classes =np.unique(yNum_test)
# roc curve for classes
fpr = {}
tpr = {}
thresh ={}
roc_auc = dict()
n_class = classes.shape[0]
for i in range(n_class):    
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test_binarized[:,i], pred_prodSys[:,i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    # plotting    
    pyplot.plot(fpr[i], tpr[i], linestyle='-', 
             label='%s vs Rest (AUC=%0.2f)'%(classes[i],roc_auc[i]))
pyplot.plot([0,1],[0,1],'b--')
pyplot.xlim([0,1])
pyplot.ylim([0,1.05])
fpr, tpr, thresholds = roc_curve(y_test_binarized[:,i], pred_prodSys[:,i])
pyplot.title('Multiclass ROC Curve')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend(loc='lower right')
plt.show()
'''