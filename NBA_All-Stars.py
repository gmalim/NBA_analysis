#!/usr/local/bin/python3 -tt
"""
Goal of the analysis is to predict the NBA All-Stars for a given year, based on All-Star selections in other years. 
This is accomplished by testing several machine learning classification algorithms and player performance statistics 
per season. The analysis in this Jupyter notebook is based on the Scikit-learn machine learning package, NBA player 
data are taken from https://www.basketball-reference.com.

Input: validation_year, prediction_year, includeadvancedstats, classifier

Author: Gordon Lim
Last Edit: 23 Mar 2018 
"""

import NBAanalysissetup # See NBAanalysissetup.py

import matplotlib.pyplot as plt
import numpy as np
import sys

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

    training_years = list(range(2015, 2019))
    
    training_years.remove(validation_year)
    training_years.remove(prediction_year)
    
    first_training_year = training_years[0]
    last_training_year  = training_years[-1]

    print("--> Validation_year = {}".format(validation_year))
    print("--> Prediction_year = {}".format(prediction_year))
    if ((first_training_year < validation_year < last_training_year) and
        (first_training_year < prediction_year < last_training_year)):
        print("--> Training years = [{}-{}] except {} and {}".format(first_training_year, last_training_year, validation_year, prediction_year))
    elif (first_training_year < validation_year < last_training_year):
        print("--> Training years = [{}-{}] except {}".format(first_training_year, last_training_year, validation_year))
    elif (first_training_year < prediction_year < last_training_year):
        print("--> Training years = [{}-{}] except {}".format(first_training_year, last_training_year, prediction_year))
    else:
        print("--> Training years = [{}-{}]".format(first_training_year, last_training_year))
    print("")

    # 1-2) Choose if you want to include advanced players statistics (e.g. PER, WS, etc.) in the model or not.
        
    includeadvancedstats = True
    
    if includeadvancedstats:
        print("--> Advanced statistics included")
    else:
        print("--> Advanced statistics not included")
    print("")
    
    # 1-3) Choose the ML classifier algorithm that you want to use:
    #
    #  1 : Logistic Regression Classifier
    #  2 : Gaussian Naive Bayes Classifier
    #  3 : Gaussian Process Classifier
    #  4 : Nearest Neighbours Classifier
    #  5 : Linear Support Vector Machine Classifier
    #  6 : Decision Tree Classifier
    #  7 : Random Forest Classifier
    #  8 : Extra Trees Classifier
    #  9 : Gradient Tree Boosting Classifier
    # 10 : Ada Boost Classifier
    # 11 : Quadratic Discriminant Analysis Classifier
    # 12 : Neural Network Classifier
        
    classifier = 5
    
    if (len(sys.argv) > 3):
        classifier_str = sys.argv[3]
        classifier = int(classifier_str)

    # Set hyper-parameters and instantiate model:
        
    if (classifier == 1):
        from sklearn import linear_model
        C = 0.4  # smaller value for C results in more regularization (to suppress noisy observations)
        p = 'l2' # penalty="l1" enables Lasso regularization, penalty="l2" enables Ridge regularization. Ridge gives Shrinkage (i.e. non-sparse coefficients), Lasso gives Sparsity (i.e. prefer simpler models)
        model = linear_model.LogisticRegression(C=C, penalty=p)
        modelname = 'Logistic Regression Classifier'
    elif (classifier == 2):
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
        modelname = 'Gaussian Naive Bayes Classifier'
    elif (classifier == 3):
        from sklearn.gaussian_process import GaussianProcessClassifier
        model = GaussianProcessClassifier() #(1.0 * RBF(1.0))
        modelname = 'Gaussian Process Classifier'
    elif (classifier == 4):
        from sklearn.neighbors import KNeighborsClassifier
        n = 10
        model = KNeighborsClassifier(n_neighbors=n)
        modelname = 'Nearest Neighbours Classifier'
    elif (classifier == 5):
        from sklearn.svm import LinearSVC                
        #from sklearn.svm import SVC                
        C = 0.05 # See Logistic Regression
        p = 'l2' # See Logistic Regression
        model = LinearSVC(C=C, penalty=p, dual=False)
        #model = SVC(kernel='linear', C=C, class_weight='balanced') #(gamma=2)
        modelname = 'Linear Support Vector Machine Classifier'
    elif (classifier == 6):
        from sklearn.tree import DecisionTreeClassifier
        m = None
        model = DecisionTreeClassifier(max_depth=m)
        modelname = 'Decision Tree Classifier'
    elif (classifier == 7):
        from sklearn.ensemble import RandomForestClassifier
        m = None
        n = 100
        f = 'auto'
        model = RandomForestClassifier(n_estimators=n, max_depth=m, max_features=f) #, class_weight='balanced'
        modelname = 'Random Forest Classifier'
    elif (classifier == 8):
        from sklearn.ensemble import ExtraTreesClassifier
        m = None
        n = 100
        model = ExtraTreesClassifier(max_depth=m, n_estimators=n, min_samples_split=2, random_state=0)
        modelname = 'Extra Trees Classifier'
    elif (classifier == 9):
        from sklearn.ensemble import GradientBoostingClassifier
        n = 100
        m = None
        l = 0.1
        model = GradientBoostingClassifier(n_estimators=n, max_depth=m, learning_rate=l)
        modelname = 'Gradient Tree Boosting Classifier'
    elif (classifier == 10):
        from sklearn.ensemble import AdaBoostClassifier
        n = 100
        l = 0.1
        model = AdaBoostClassifier(n_estimators=n, learning_rate=l)
        modelname = 'Ada Boost Classifier'
    elif (classifier == 11):
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        model = QuadraticDiscriminantAnalysis()
        modelname = 'Quadratic Discriminant Analysis Classifier'
    elif (classifier == 12):
        from sklearn.neural_network import MLPClassifier
        a = 1
        model = MLPClassifier(alpha=a)
        modelname = 'Neural Network Classifier'
    else:
        print("That number does not correspond to an implemented classifier - EXIT")
    
    print("--> Selected ML classifier =", modelname)
    print("")
    
    # 2-1) Load NBA player data from csv files in data directory:

    df_training, df_validation, df_prediction = NBAanalysissetup.loaddata_allyears(prediction_year, \
                                                                                   validation_year, \
                                                                                   training_years, \
                                                                                   includeadvancedstats)

    print("")
    
    # 2-2) Prepare data and create features (X) and target (y) dataframes needed for sklearn routines:

    # Remove players which have NAN(s):

    df_training   = df_training.  dropna()
    df_validation = df_validation.dropna()
    df_prediction = df_prediction.dropna()

    '''
    # Split data in training and test sets:

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_NBAplayers, y_NBAplayers, test_size=0.2, random_state=1)

    print("--- All data:  ", X_NBAplayers.shape, y_NBAplayers.shape)
    print("--- Train data:", X_train.shape,      y_train.shape)
    print("--- Test data: ", X_test.shape,       y_test.shape)
    '''

    # Remove AS feature (binary variable indicating All-Star selection) since this is what we want to predict
    # Remove YEAR feature since this is only used for the cross-validation scheme
    # Remove features which do not have predictive power (e.g. Player name, Team name, etc.) 
    # Remove features which are correlated with other features (e.g. FGA, 3PA, etc.)

    X_training    = df_training.  drop(['AS', 'YEAR', 'Player', 'Pos', 'Tm', 'Age', 'FGA/G', '3PA/G', '2PA/G', 'FTA/G', 'TRB/G'], axis=1)
    X_validation  = df_validation.drop(['AS', 'YEAR', 'Player', 'Pos', 'Tm', 'Age', 'FGA/G', '3PA/G', '2PA/G', 'FTA/G', 'TRB/G'], axis=1)
    X_prediction  = df_prediction.drop(['AS', 'YEAR', 'Player', 'Pos', 'Tm', 'Age', 'FGA/G', '3PA/G', '2PA/G', 'FTA/G', 'TRB/G'], axis=1)

    if includeadvancedstats:

        X_training    = X_training.  drop(['WS'], axis=1)
        X_validation  = X_validation.drop(['WS'], axis=1)
        X_prediction  = X_prediction.drop(['WS'], axis=1)
    
    # The target is the AS feature:
    
    y_training    = df_training  ['AS']
    y_validation  = df_validation['AS']
    y_prediction  = df_prediction['AS']

    print("--> Training   data set      : # of players = {}, # of features = {}".format(X_training.shape[0], X_training.shape[1]))
    print("--> Validation data set {} : # of players = {:4}, # of features = {}".format(validation_year, X_validation.shape[0], X_validation.shape[1]))
    print("--> Prediction data set {} : # of players = {:4}, # of features = {}".format(prediction_year, X_prediction.shape[0], X_prediction.shape[1]))
    print("")

    print("--> Model features : ", list(X_training.columns))
    print("")

    # 3) Cross-validate the model using training data and the LeaveOneGroupOut cross-validation scheme, and calculate some model scores:

    print("--> Apply LeaveOneGroupOut cross-validation scheme to training data...")
    print("")
    
    from sklearn.model_selection import cross_validate, LeaveOneGroupOut

    logo = LeaveOneGroupOut()

    cv_groups = df_training['YEAR'] # Use YEAR feature to define groups in LeaveOneGroupOut cross-validation scheme

    cv_logo = logo.split(X_training, y_training, groups=cv_groups)

    scoring_list = ['precision', 'recall', 'f1', 'accuracy', 'roc_auc']

    scores = cross_validate(model, X_training, y_training, cv=cv_logo, scoring=scoring_list)

    # NOTE: If n_jobs=-1 in cross_validate, Python will timeout if training set is too large
    #       (i.e. if traning set including advanced statistics contains more than 7 years)
    #       This is only the case when using python or ipython, not with jupyter notebook)

    print("--> Cross-val years  :", ["{:5d}".format(yr) for yr in training_years])
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

    y_valtrue  = y_validation.tolist()

    from sklearn.metrics import confusion_matrix

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

    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

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

    from sklearn.metrics import roc_curve, precision_recall_curve, auc

    decisionfunctionclassifiers = [1, 5] # LogisticRegression, SVC

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
            print("----> Model coefficient {:5} = {:>6.3f}".format(name, coef))
        print("")
        
    if hasattr(model, "feature_importances_"):
        print("--> Feature importances: ")
        print("")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        for i in range(X_training.shape[1]):
            print("----> Feature importance ({:>2}) {:5} : {:.3f}".format(i + 1, X_training.columns[indices[i]], importances[indices[i]]))
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

    note1 = " (Answers the question: Out of all true All-Stars, how many are predicted?)"
    note2 = " (Answers the question: Out of all predicted All-Stars, how many are true?)"
    note3 = " (i.e. the harmonic mean of Precision and Recall)"
    note4 = " (Answers the question: Out of all players, how many are correctly predicted?)"
    
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
    from sklearn.metrics import classification_report
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
        
    import operator
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
            
    plt.show()
        
    return 0

if __name__ == '__main__':
    main()
