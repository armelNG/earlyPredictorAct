# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 22:40:54 2024

@author: NGATCH04
"""
import time
import matplotlib.pylab
import numpy as np
import pandas as pdact
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import randint
from sklearn import metrics
import pandas as pdact

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
def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")

#--------------------------------------------------------------------------------------------

# RandomSearch Grid
rf = RandomForestClassifier()
n_estimators = randint(50,5000)
max_depth = randint(1,200)
max_features = ['log2','sqrt','none']
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
rand_search = RandomizedSearchCV(rf, param_distributions = random_grid, n_iter=10, cv=3, verbose=2, random_state=42, scoring= 'neg_mean_absolute_error')
rand_search.fit(x_app, yNum_app)
print(rand_search.cv_results_)
best_rf = rand_search.best_estimator_
print('Best hyperparameters:',  rand_search.best_params_)

grid_final_accuracy = evaluate_model(best_rf, x_app, yNum_app, x_test, yNum_test)
print(grid_final_accuracy)

best_rf.fit(x_app, yNum_app)
predictions = best_rf.predict(x_test)
dfpredictions = pdact.DataFrame(predictions, columns=['Results'])


# Matrices de confusion
# Create the confusion matrix
cm = metrics.confusion_matrix(yNum_test, predictions)
metrics.ConfusionMatrixDisplay(confusion_matrix=cm).plot();
plt.show()

# Importance des variables
print(pdact.DataFrame(best_rf.feature_importances_, index = x_app.columns, columns = ["importance"]).sort_values("importance", ascending = False))
plt.figure(figsize=(15,10))
plt.bar([i for i in range(len(best_rf.feature_importances_))], best_rf.feature_importances_, tick_label=x_app.columns)
plt.show()

# Plot results of algorithm
# plot_results(rand_search)

# Plot courbe d'apprentissage
#plot_learning_curves(best_rf, x_app, yNum_app)

#------------------------------------------------------------------------------------------------

# Generation du fichier des résultats
R = pdact.DataFrame()
for i in range(1, 30):
    best_rf.fit(x_app.iloc[:,0:i], y_app)
    result = best_rf.predict(x_test.iloc[:,0:i])
    print(result)
    R[i] = result
print(R)
R.to_csv('./results/matrices/act/ResultAct_rf.csv', index=True)

#------------------------------------------------------------------------------------------------