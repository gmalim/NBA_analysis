#!/usr/local/bin/python3 -tt
"""
The goal of this analysis is to predict the NBA All-Stars for a given year, based on All-Star selections
in other years. This is accomplished by applying several machine learning classification algorithms to 
player performance data. The analysis is based on the Scikit-learn machine learning package, NBA player 
data are taken from https://www.basketball-reference.com/.

Input: validation_year, prediction_year, includeadvancedstats, classifier

Author: Gordon Lim
Last Edit: 31 Mar 2018 
"""

import NBAanalysissetup # See NBAanalysissetup.py

import matplotlib.pyplot as plt
import numpy as np
import operator
import sys

from IPython.display import display, HTML

from sklearn.linear_model          import LogisticRegression
from sklearn.neighbors             import KNeighborsClassifier
from sklearn.svm                   import LinearSVC, SVC
from sklearn.tree                  import DecisionTreeClassifier
from sklearn.ensemble              import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network        import MLPClassifier
from sklearn.naive_bayes           import GaussianNB
from sklearn.gaussian_process      import GaussianProcessClassifier

from sklearn.model_selection import cross_validate, LeaveOneGroupOut

from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

def main():

    # 1-1) Choose the year you want to use for validation, and the year you want to predict (both in range 2000-2018).
    #      The years in range 2000-2018 that are not selected for validation and prediction are used to train the model:

    validation_year = 2017
    if (len(sys.argv) > 1):
        validation_year_str = sys.argv[1]
        validation_year = int(validation_year_str)
        
    prediction_year = 2018
    if (len(sys.argv) > 2):
        prediction_year_str = sys.argv[2]
        prediction_year = int(prediction_year_str)

    print("--> Validation_year = {}".format(validation_year))
    print("--> Prediction_year = {}".format(prediction_year))
    
    first_year = 2000 # First year for which data has been scraped
    last_year  = 2018 # Last  year for which data has been scraped
    
    training_years = list(range(first_year, last_year+1))
    
    training_years.remove(validation_year)
    training_years.remove(prediction_year)
    
    first_training_year = training_years[0]
    last_training_year  = training_years[-1]
    
    if (first_training_year < validation_year < last_training_year) and (first_training_year < prediction_year < last_training_year):
        print("--> Training years  = {}-{} except {} and {}".format(first_training_year, last_training_year, validation_year, prediction_year))
    elif (first_training_year < validation_year < last_training_year):
        print("--> Training years  = {}-{} except {}".format(first_training_year, last_training_year, validation_year))
    elif (first_training_year < prediction_year < last_training_year):
        print("--> Training years  = {}-{} except {}".format(first_training_year, last_training_year, prediction_year))
    else:
        print("--> Training years  = {}-{}".format(first_training_year, last_training_year))
    print("")

    # 1-2) Choose if you want to include advanced players statistics (e.g. PER, WS, etc.) in the model or not, and
    #      choose the minimum number of games a player has to have played in a season to be included in the analysis:

    includeadvancedstats = True # Enter True or False

    if includeadvancedstats:
        print("--> Advanced statistics included")
    else:
        print("--> Advanced statistics not included")
        
    min_num_games = 10 # Enter any number between 0 and 82

    print("--> Minimum number of games for each player =", min_num_games)
    print("")
    
    # 1-3) Choose the ML classifier algorithm that you want to use:
    #
    #  1 : Logistic Regression Classifier
    #  2 : Nearest Neighbours Classifier
    #  3 : Linear Support Vector Machine Classifier
    #  4 : Decision Tree Classifier
    #  5 : Random Forest Classifier
    #  6 : Extra Trees Classifier
    #  7 : Gradient Tree Boosting Classifier
    #  8 : Ada Boost Classifier
    #  9 : Quadratic Discriminant Analysis Classifier
    # 10 : Neural Network Classifier
    # 11 : Gaussian Naive Bayes Classifier
    # 12 : Gaussian Process Classifier
        
    classifier = 1
    
    if (len(sys.argv) > 3):
        classifier_str = sys.argv[3]
        classifier = int(classifier_str)

    # Set hyper-parameters (set random_state to specific seed value where applicable to tune hyper-parameters) and instantiate model:
        
    rseed = 666

    if (classifier == 1):
        C = 100  # smaller value for C results in more regularization (in case you have noisy observations)
        p = 'l2' # penalty="l1" enables Lasso regularization, penalty="l2" enables Ridge regularization. Ridge gives Shrinkage (i.e. non-sparse coefficients), Lasso gives Sparsity (i.e. prefer simpler models)
        model = LogisticRegression(C=C, penalty=p, random_state=rseed)
        modelname = 'Logistic Regression Classifier'
    elif (classifier == 2):
        n_n = 10
        w = 'uniform'
        model = KNeighborsClassifier(n_neighbors=n_n, weights=w)
        modelname = 'Nearest Neighbours Classifier'
    elif (classifier == 3):
        C = 1
        p = 'l2'
        model = LinearSVC(dual=False, C=C, penalty=p, random_state=rseed) #class_weight='balanced'
        #model = SVC(kernel='linear', probability=True, C=C, class_weight='balanced', random_state=rseed)
        modelname = 'Linear Support Vector Machine Classifier'
    elif (classifier == 4):
        m_d = None
        m_f = 'auto'
        model = DecisionTreeClassifier(max_depth=m_d, max_features=m_f, class_weight='balanced', random_state=rseed)
        modelname = 'Decision Tree Classifier'
    elif (classifier == 5):
        n_e = 100
        m_d = None  # i.e. nodes are expanded until leafs are pure 
        m_f = 'auto' # The number of features to consider when looking for the best split. 'auto' => max_features=sqrt(n_features) 
        model = RandomForestClassifier(n_estimators=n_e, max_depth=m_d, max_features=m_f, class_weight='balanced', random_state=rseed)
        modelname = 'Random Forest Classifier'
    elif (classifier == 6):
        n_e = 100
        m_d = None
        m_f = 'auto'
        model = ExtraTreesClassifier(n_estimators=n_e, max_depth=m_d, max_features=m_f, class_weight='balanced', random_state=rseed)
        modelname = 'Extremely Randomized Trees Classifier'
    elif (classifier == 7):
        n_e = 100
        m_d = 3
        l_l = 0.1
        model = GradientBoostingClassifier(n_estimators=n_e, max_depth=m_d, learning_rate=l_l, random_state=rseed)
        modelname = 'Gradient Boost Classifier'
    elif (classifier == 8):
        n_e = 100
        l_l = 0.1
        model = AdaBoostClassifier(n_estimators=n_e, learning_rate=l_l, random_state=rseed)
        modelname = 'AdaBoost Classifier'
    elif (classifier == 9):
        model = QuadraticDiscriminantAnalysis()
        modelname = 'Quadratic Discriminant Analysis Classifier'
    elif (classifier == 10):
        a = 0.0001 # L2 penalty (regularization term) parameter
        model = MLPClassifier(alpha=a, random_state=rseed)
        modelname = 'Neural Network Classifier'
    elif (classifier == 11):
        model = GaussianNB()
        modelname = 'Gaussian Naive Bayes Classifier'
    elif (classifier == 12): # SLOWWWWWWWWW
        model = GaussianProcessClassifier(random_state=rseed)
        modelname = 'Gaussian Process Classifier'
    else:
        print("That number does not correspond to an implemented classifier - EXIT")
    
    print("--> Selected classifier =", modelname)
    print("--> Model parameters : ", model.get_params())
    print("")
    
    # 2-1) Load NBA player data from csv files in data directory:

    df_training, df_validation, df_prediction = NBAanalysissetup.loaddata_allyears(prediction_year, \
                                                                                   validation_year, \
                                                                                   training_years, \
                                                                                   includeadvancedstats)

    print("")
    
    # 2-2) Remove players which have played less than min_num_games number of games:

    print("--> # of players in   training set =", df_training  .shape[0])
    print("--> # of players in validation set =", df_validation.shape[0])
    print("--> # of players in prediction set =", df_prediction.shape[0])
    print("")
    
    df_training   = df_training  [df_training  ['G'] >= min_num_games]
    df_validation = df_validation[df_validation['G'] >= min_num_games]
    df_prediction = df_prediction[df_prediction['G'] >= min_num_games]
    
    print("--> # of players in   training set =", df_training  .shape[0])
    print("--> # of players in validation set =", df_validation.shape[0])
    print("--> # of players in prediction set =", df_prediction.shape[0])
    print("")

    # 2-3) NaN handling:

    print("--> # of players with NaNs in   training set =", df_training  .shape[0] - df_training  .dropna().shape[0])
    print("--> # of players with NaNs in validation set =", df_validation.shape[0] - df_validation.dropna().shape[0])
    print("--> # of players with NaNs in prediction set =", df_prediction.shape[0] - df_prediction.dropna().shape[0])
    print("")

    # Replace NaNs with 0s in the following columns:

    df_training  [['FG%', '3P%', '2P%', 'FT%', 'eFG%']] = df_training  [['FG%', '3P%', '2P%', 'FT%', 'eFG%']].fillna(value=0)
    df_validation[['FG%', '3P%', '2P%', 'FT%', 'eFG%']] = df_validation[['FG%', '3P%', '2P%', 'FT%', 'eFG%']].fillna(value=0)
    df_prediction[['FG%', '3P%', '2P%', 'FT%', 'eFG%']] = df_prediction[['FG%', '3P%', '2P%', 'FT%', 'eFG%']].fillna(value=0)
    
    if (includeadvancedstats):
        df_training  [['TS%', '3PAr', 'FTr']] = df_training  [['TS%', '3PAr', 'FTr']].fillna(value=0)
        df_validation[['TS%', '3PAr', 'FTr']] = df_validation[['TS%', '3PAr', 'FTr']].fillna(value=0)
        df_prediction[['TS%', '3PAr', 'FTr']] = df_prediction[['TS%', '3PAr', 'FTr']].fillna(value=0)
    
    print("--> # of players with NaNs in   training set =", df_training  .shape[0] - df_training  .dropna().shape[0])
    print("--> # of players with NaNs in validation set =", df_validation.shape[0] - df_validation.dropna().shape[0])
    print("--> # of players with NaNs in prediction set =", df_prediction.shape[0] - df_prediction.dropna().shape[0])
    
    # Remove remaining players with NaNs, if necessary:

    if (df_training.shape[0] - df_training.dropna().shape[0] != 0):
        print("")
        #print("--> Players in training set with NaNs:")
        #print(df_training[df_training.isnull().any(axis=1)])
        #print("")
        df_training.dropna(inplace=True)
        print("--> # of players with NaNs in   training set =", df_training.shape[0] - df_training.dropna().shape[0])

    if (df_validation.shape[0] - df_validation.dropna().shape[0] != 0):
        print("")
        #print("--> Players in validation set with NaNs:")
        #print(df_validation[df_validation.isnull().any(axis=1)])
        #print("")
        df_validation.dropna(inplace=True)
        print("--> # of players with NaNs in validation set =", df_validation.shape[0] - df_validation.dropna().shape[0])

    if (df_prediction.shape[0] - df_prediction.dropna().shape[0] != 0):
        print("")
        #print("--> Players in prediction set with NaNs:")
        #print(df_prediction[df_prediction.isnull().any(axis=1)])
        #print("")
        df_prediction.dropna(inplace=True)
        print("--> # of players with NaNs in prediction set =", df_prediction.shape[0] - df_prediction.dropna().shape[0])

    print("")

    # 2-4) Print overview of All-Stars:

    for year in training_years:
        n_allstars = df_training[(df_training['YEAR'] == year) & (df_training['AS'] > 0.5)].shape[0]
        n_total    = df_training[(df_training['YEAR'] == year)].shape[0]
        print("--> Number if All-Stars in   training year {}: {} out of {} players".format(year, n_allstars, n_total))
    
    n_allstars = df_validation[df_validation['AS'] > 0.5].shape[0]
    n_total    = df_validation.shape[0]
    print("--> Number if All-Stars in validation year {}: {} out of {} players".format(validation_year, n_allstars, n_total))
    
    n_allstars = df_prediction[df_prediction['AS'] > 0.5].shape[0]
    n_total    = df_prediction.shape[0]
    print("--> Number if All-Stars in prediction year {}: {} out of {} players".format(prediction_year, n_allstars, n_total))

    print("")
    '''
    print("--> All-Stars {} :".format(validation_year))
    #tmp = df_training[(df_training['YEAR'] == 2016) & (df_training['AS'] > 0.5)]
    tmp = df_validation[df_validation['AS'] > 0.5]
    display(HTML(tmp.to_html()))
    '''
    
    # 2-5) Prepare data and create features (X) and target (y) dataframes needed for Scikit-learn methods:

    '''
    # Split data in training and test sets:

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_NBAplayers, y_NBAplayers, test_size=0.2, random_state=1)

    print("--- All data:  ", X_NBAplayers.shape, y_NBAplayers.shape)
    print("--- Train data:", X_train.shape,      y_train.shape)
    print("--- Test data: ", X_test.shape,       y_test.shape)
    '''
    
    # Features dataframes (AS and YEAR are dropped: AS is the target variable, YEAR is only used for cross-validation):

    X_training   = df_training.  drop(['AS', 'YEAR'], axis=1)
    X_validation = df_validation.drop(['AS', 'YEAR'], axis=1)
    X_prediction = df_prediction.drop(['AS', 'YEAR'], axis=1)
    
    X_list = [X_training, X_validation, X_prediction]
    
    for X in X_list:
    
        # Remove features:

        X.drop(['Player', 'Pos', 'Tm', 'Age',                                   # No predictive power
                'FG', 'FGA', 'FG%', '3P%', '2P%', 'FT%', 'eFG%', 'TRB', 'PTS'], # Correlated with other features
               axis=1, inplace=True)

        if includeadvancedstats:
            X.drop(['TS%', '3PAr', 'FTr', 'TRB%', 'WS', 'WS/48', 'BPM',         # Correlated with other features 
                    'ORB', 'DRB', 'AST', 'TOV', 'STL', 'BLK'],                  # Correlated with other features
                   axis=1, inplace=True) 
    
        # Replace GS by GS/G, and MP by MP/48 and drop G:

        X['GS'] = X['GS'].div(X['G'].values, axis=0)
        X['MP'] = X['MP'].div(48,            axis=0)
        X.rename(columns={'GS': 'GS/G', 'MP': 'MP/48'}, inplace=True)
        X.drop(['G'], axis=1, inplace=True)
        
        # Scale total-type features by MP/48:

        X[['3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'PF']] = X[['3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'PF']].div(X['MP/48'].values, axis=0)
        X.rename(columns={'3P': '3P/48', '3PA': '3PA/48', 'FT': 'FT/48', 'FTA': 'FTA/48', 
                          '2P': '2P/48', '2PA': '2PA/48', 'PF': 'PF/48'}, inplace=True)
    
        if includeadvancedstats:
            X[['OWS', 'DWS']] = X[['OWS', 'DWS']].div(X['MP/48'].values, axis=0)
            X.rename(columns={'OWS': 'OWS/48', 'DWS': 'DWS/48'}, inplace=True)
        else:
            X[['ORB', 'DRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']] = X[['ORB', 'DRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']].div(X['MP/48'].values, axis=0)
            X.rename(columns={'ORB': 'ORB/48', 'DRB': 'DRB/48', 'AST': 'AST/48', 'PF': 'PF/48',
                              'BLK': 'BLK/48', 'TOV': 'TOV/48', 'STL': 'STL/48'}, inplace=True)
    
    # Target dataframes (target = AS, a binary variable introduced to indicate All-Star status):

    y_training   = df_training  ['AS']
    y_validation = df_validation['AS']
    y_prediction = df_prediction['AS']
    
    print("--> Training   data set      : # of players = {:4}, # of features = {}".format(X_training.shape[0], X_training.shape[1]))
    print("--> Validation data set {} : # of players = {:4}, # of features = {}".format(validation_year, X_validation.shape[0], X_validation.shape[1]))
    print("--> Prediction data set {} : # of players = {:4}, # of features = {}".format(prediction_year, X_prediction.shape[0], X_prediction.shape[1]))
    print("")
    
    print("--> Model features : ", list(X_training.columns))
    print("")
    
    # 3) Cross-validate the model using training data and the LeaveOneGroupOut cross-validation scheme in which a group is defined as a single NBA season,
    #    and calculate some model scores:

    print("--> Apply LeaveOneGroupOut cross-validation scheme to training data...")
    print("")
    
    logo = LeaveOneGroupOut()
        
    cv_groups = df_training['YEAR'] # Players in the same group (i.e. a single NBA season) have identical YEAR variables

    cv_logo = logo.split(X_training, y_training, groups=cv_groups)

    scoring_list = ['precision', 'recall', 'f1', 'accuracy', 'roc_auc']
    
    scores = cross_validate(model, X_training, y_training, cv=cv_logo, scoring=scoring_list) #, n_jobs=-1)

    # NOTE: If n_jobs=-1 in cross_validate, Python will timeout if training set is too large
    #       (i.e. if traning set including advanced statistics contains more than 7 years)
    #       This is only the case when using python or ipython, not with jupyter notebook)

    print("--> Cross-val years  :", ["{:5d}".format(yr) for yr in training_years])
    print("")
    print("--> Precision scores :", ["{:5.2f}".format(i) for i in scores['test_precision']])
    print("--> Recall    scores :", ["{:5.2f}".format(i) for i in scores['test_recall'   ]])
    print("--> F1        scores :", ["{:5.2f}".format(i) for i in scores['test_f1'       ]])
    print("--> Accuracy  scores :", ["{:5.2f}".format(i) for i in scores['test_accuracy' ]])
    print("--> ROC-AUC   scores :", ["{:5.2f}".format(i) for i in scores['test_roc_auc'  ]])
    print("")
    print("--> Precision score : {:5.1%} +/- {:5.1%}".format(np.mean(scores['test_precision']), np.std(scores['test_precision'])))
    print("--> Recall score    : {:5.1%} +/- {:5.1%}".format(np.mean(scores['test_recall'   ]), np.std(scores['test_recall'   ])))
    print("--> F1 score        : {:5.1%} +/- {:5.1%}".format(np.mean(scores['test_f1'       ]), np.std(scores['test_f1'       ])))
    print("--> Accuracy score  : {:5.1%} +/- {:5.1%}".format(np.mean(scores['test_accuracy' ]), np.std(scores['test_accuracy' ])))
    print("--> ROC-AUC score   : {:5.1%} +/- {:5.1%}".format(np.mean(scores['test_roc_auc'  ]), np.std(scores['test_roc_auc'  ])))
    print("")
    
    # 4-1) Fit model to training data, use fitted model to predict validation data and calculate the corresponding confusion matrix and some model scores:

    model.fit(X_training, y_training)     # Fit model to training data

    y_model = model.predict(X_validation) # Predict validation data

    y_valtrue = y_validation.tolist()
    
    CM = confusion_matrix(y_valtrue, y_model) # defined as: rows -> true, columns -> prediction
    
    print("--> Confusion matrix {}:".format(validation_year))
    print(CM)
    print("")
    
    TN = CM[0,0] # defined as: 0 = negative, 1 = positive
    FN = CM[1,0] # defined as: 0 = negative, 1 = positive
    FP = CM[0,1] # defined as: 0 = negative, 1 = positive
    TP = CM[1,1] # defined as: 0 = negative, 1 = positive
    
    TOT = TP + FP + FN + TN
    
    print("--> TP = {}, FP = {}, FN = {}, TN = {}".format(TP, FP ,FN, TN))
    print("")
    print("--> True  Positive Rate i.e. Recall   (TP/(TP+FN)) = {:5.1%}".format(TP/(TP+FN)))
    print("--> False Positive Rate i.e. Fall-Out (FP/(FP+TN)) = {:5.1%}".format(FP/(FP+TN)))
    print("")
    
    precision = precision_score(y_validation, y_model)
    recall    = recall_score   (y_validation, y_model)
    f1        = f1_score       (y_validation, y_model)
    accuracy  = accuracy_score (y_validation, y_model)
    roc_auc   = roc_auc_score  (y_validation, y_model)
    
    print("--> Precision score : {:.1%}".format(precision))
    print("--> Recall score    : {:.1%}".format(recall   ))
    print("--> F1 score        : {:.1%}".format(f1       ))
    print("--> Accuracy score  : {:.1%}".format(accuracy ))
    print("--> ROC-AUC score   : {:.1%}".format(roc_auc  ))
    print("")
    
    # 4-2) Calculate ROC and PR curves using validation data:

    decisionfunctionclassifiers = [1, 3] # LogisticRegression, SVC

    if (classifier in decisionfunctionclassifiers):
        y_score = model.decision_function(X_validation)
    else:
        y_score = model.predict_proba(X_validation)

    if (classifier in decisionfunctionclassifiers):
        fpr, tpr, thresholds = roc_curve(y_validation, y_score)       # to be used when y_score is calculated using decision_function method
    else:
        fpr, tpr, thresholds = roc_curve(y_validation, y_score[:, 1]) # to be used when y_score is calculated using predict_proba method

    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='red', lw=2, label='{} (AUC={:.2f})'.format(modelname, roc_auc))
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', label='Random classifier')
    plt.xlabel('False Positive Rate i.e. Fall-Out')
    plt.ylabel('True Positive Rate i.e. Recall')
    plt.title('Receiver Operating Characteristic curve {}'.format(validation_year))
    plt.legend(loc="lower right")
    #plt.text(0.65, 0.3, r"ROC-AUC = {:.2f}".format(roc_auc), color='red')
    plt.grid(True)

    if (classifier in decisionfunctionclassifiers):
        precision, recall, _ = precision_recall_curve(y_validation, y_score)       # to be used when y_score is calculated using decision_function method
    else:
        precision, recall, _ = precision_recall_curve(y_validation, y_score[:, 1]) # to be used when y_score is calculated using predict_proba method
    
    pr_auc = auc(recall, precision)

    plt.figure()
    plt.plot(recall, precision, color='green', lw=2, label='{} (AUC={:.2f})'.format(modelname, pr_auc))
    plt.plot([0, 1], [1, 0], color='black', linestyle='--', label='Random classifier')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve {}'.format(validation_year))
    plt.legend(loc="lower left")
    #plt.text(0.05, 0.3, r"PR-AUC = {:.2f}".format(pr_auc), color='green')
    plt.grid(True)

    # 4-3) Calculate the feature coefficients and importances of the fitted model, if applicable:

    if hasattr(model, "coef_"):
        print("--> Model coefficients: ")
        print("")
        for name, coef in zip(X_training.columns, model.coef_.ravel()):
            print("----> Model coefficient {:6} = {:>6.3f}".format(name, coef))
        print("")
        
    if hasattr(model, "feature_importances_"):
        print("--> Feature importances: ")
        print("")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        for i in range(X_training.shape[1]):
            print("----> Feature importance ({:>2}) {:6} : {:.3f}".format(i + 1, X_training.columns[indices[i]], importances[indices[i]]))
        print("")
        
    # 5-1) Use fitted model to predict the NBA All-Stars in prediction_year, and calculate the corresponding confusion matrix:
        
    y_model = model.predict(X_prediction) # Use fitted model on prediction data

    y_true  = y_prediction.tolist()

    CM = confusion_matrix(y_true, y_model) # defined as: rows -> true, columns -> prediction

    print("--> Confusion matrix {}:".format(prediction_year))
    print(CM)
    print("")

    TN = CM[0,0] # defined as: 0 = negative, 1 = positive
    FN = CM[1,0] # defined as: 0 = negative, 1 = positive
    FP = CM[0,1] # defined as: 0 = negative, 1 = positive
    TP = CM[1,1] # defined as: 0 = negative, 1 = positive

    TOT = TP + FP + FN + TN

    print("--> TP = {}, FP = {}, FN = {}, TN = {}".format(TP, FP ,FN, TN))
    print("")

    note1 = " (Answers the question: How many predicted All-Stars are true All-Stars?)"
    note2 = " (Answers the question: How many true All-Stars have been predicted?)"
    note3 = " (i.e. the harmonic mean of Precision and Recall)"
    note4 = " (Answers the question: How many All-Stars and non-All-Stars have been correctly predicted?)"
    
    print("--> Precision (TP/(TP+FP)) = {:5.1%}".format(TP/(TP+FP))         + note1)
    print("--> Recall    (TP/(TP+FN)) = {:5.1%}".format(TP/(TP+FN))         + note2)
    print("--> F1 score               = {:5.1%}".format(2*TP/(2*TP+FP+FN))  + note3)
    print("--> Accuracy ((TP+TN)/TOT) = {:5.1%}".format((TP+TN)/TOT)        + note4) 
    print("")
    
    np.set_printoptions(precision=2)
    class_names = ['Non-All-Star','All-Star']

    plt.figure()
    NBAanalysissetup.plot_confusion_matrix(CM, classes=class_names,
                                           title='Confusion matrix {}'.format(prediction_year))

    #plt.figure()
    #NBAanalysissetup.plot_confusion_matrix(CM, classes=class_names, normalize=True,
    #                                       title='Normalized confusion matrix')   
    
    '''
    plt.figure()
    import seaborn as sns; sns.set()
    sns.heatmap(CM.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('true label')
    plt.ylabel('predicted label');
    '''

    # 5-2) Print classification report:

    print("--> Classification report for {}:".format(prediction_year))
    print("")
    class_names = ['non-All-Stars (true)', 'All-Stars (true)']
    print(classification_report(y_true, y_model, target_names=class_names))
    print("")
    
    # 5-3) Check which players are All-Stars according to the model:

    counts = [0, 0, 0, 0]

    snubbed      = []
    deserved     = []
    questionable = []
    
    for i in range(0, len(y_model)):
        if ((y_true[i] == 0) and (y_model[i] == 0)):    # TN
            counts[0] += 1
        elif ((y_true[i] == 0) and (y_model[i] == 1)):  # FP
            counts[1] += 1
            snubbed.append(df_prediction.iat[i,0])              # 0-th column in df is player name
        elif ((y_true[i] == 1) and (y_model[i] == 0)):  # FN
            counts[2] += 1
            questionable.append(df_prediction.iat[i,0])         # 0-th column in df is player name
        else:                                           # TP
            counts[3] += 1
            deserved.append(df_prediction.iat[i,0])             # 0-th column in df is player name
        
    print("--> # of     All-Stars predicted to be     All-Stars = {:>3} (TP)".format(counts[3]))
    print("--> # of non-All-Stars predicted to be     All-Stars = {:>3} (FP)".format(counts[1]))
    print("--> # of     All-Stars predicted to be non-All-Stars = {:>3} (FN)".format(counts[2]))
    print("--> # of non-All-Stars predicted to be non-All-Stars = {:>3} (TN)".format(counts[0]))
    print("")
    print("--> Deserved true All-Stars:     ", deserved)
    print("")
    print("--> Questionable true All-Stars: ", questionable)
    print("")
    print("--> Snubbed non-All-Stars:       ", snubbed)
    print("")
    
    # 5-4) List all NBA players in prediction_year according to their model scores:

    print("--> Model scores for all players in {}:".format(prediction_year))
    print("")
    
    if (classifier in decisionfunctionclassifiers):
        y_score = model.decision_function(X_prediction)
    else:
        y_score = model.predict_proba(X_prediction)

    player_score_dict = {}
    player_AS_dict    = {}
    
    if includeadvancedstats:
        AS_index = 49
    else:
        AS_index = 29
    
    for i in range(0, len(y_model)):
        if (classifier in decisionfunctionclassifiers):
            player_score_dict[df_prediction.iat[i,0]] = y_score[i].ravel()[0]
        else:
            player_score_dict[df_prediction.iat[i,0]] = y_score[i].ravel()[1]
        if df_prediction.iat[i,AS_index] > 0.5:
            status = 'All-Star'
        else:
            status = 'Non-All-Star'
        player_AS_dict[df_prediction.iat[i,0]] = status
        
    sorted_player_score_dict = sorted(player_score_dict.items(), key=operator.itemgetter(1), reverse=True)
    
    counter = 0
    printed_line = False
    for key, value in dict(sorted_player_score_dict).items():
        counter += 1
        if (classifier in decisionfunctionclassifiers):
            if (value < 0 and not printed_line):
                print("**********************************************************")
                printed_line = True
        else:
            if (value < 0.5 and not printed_line):
                print("**********************************************************")
                printed_line = True
        print("----> {:3}: {:24} = {:.3f} ({})".format(counter, key, value, player_AS_dict[key]))

    print("")

    # 5-5) Print prediction features of any particular player:

    player_name = 'LeBron James'

    for i in range(0, len(X_prediction)):
        if (df_prediction.iat[i,0] == player_name):
            player_index = i
        
    print("--> Classification input for", player_name, ":")
    print(X_prediction.iloc[player_index])
    
    plt.show()
        
    return 0

if __name__ == '__main__':
    main()
