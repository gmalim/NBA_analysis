# Predicting the NBA MVP with Machine Learning

The goal of this analysis is to predict the NBA All-Stars for a given year, based on NBA player data and MVP voting statistics in other years. This is accomplished by applying several machine learning regression algorithms on NBA player performance data. The analysis is based on the [Scikit-learn](http://scikit-learn.org) machine learning package for Python. NBA player data are taken from [basketball-reference.com](https://www.basketball-reference.com), data from 2010-2018 is included in the **data** directory of this repository. Data from other years can be obtained by using [*Basketball_Reference_scraper.py*](Basketball_Reference_scraper.py).  

## Analysis

The analysis is presented as a [Jupyter Notebook](NBA_MVP.ipynb). The outline of the analysis is summarized in the following:

### 1. Import external modules and libraries

- [NumPy](http://www.numpy.org)
- [Pandas](https://pandas.pydata.org)
- [Scikit-learn](http://scikit-learn.org)
- [XGBoost](http://xgboost.readthedocs.io/en/latest/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)

### 2. User input

- Choose the year you want to predict, between 2010 and 2018. The years that are not selected are used for cross-validation and training of the ML algorithms.
- Choose whether you want to include advanced player statistics (e.g. *PER*, *VORP*, etc.) in the analysis or not.
- Choose the minimum number of games a player has to have played per season to be included in the analysis.

### 3. NBA player data

- Data loading: NBA player data from 2010-2018 from [basketball-reference.com](https://www.basketball-reference.com) have been saved as csv-files in the **data** directory using the scraper functions in [*NBAanalysissetup.py*](NBAanalysissetup.py).
- Data preparation (feature selection, *NaN* handling, etc.).
- Features included in this analysis: *G, GS/G, MP/G, 3P, 3PA, 2P, 2PA, FT, FTA, PF, PER, ORB%, DRB%, AST%, STL%, BLK%, TOV%, USG%, OWS, DWS, OBPM, DBPM, VORP, TW/82*. (Definitions can be found [here](https://www.basketball-reference.com/about/glossary.html)).
- Feature scaling as required by various ML algorithms.

### 4. Supervised Learning

- Selection of various popular ML regression algorithms:
	- Nearest Neighbours Regressor
	- Support Vector Machine Regressor
	- Neural Network Regressor
	- Random Forest Regressor
	- Extra Randomized Trees Regressor
	- Gradient Boosted Decision Tree Regressor
	- Adaptive Boosted (AdaBoost) Decision Tree Regressor
	- Extreme Gradient Boosted (XGBoost) Decision Tree Regressor
- Hyper-parameter tuning and instantiation of all models.

### 5. Cross-validation 

- All regressors are cross-validated by using training data and the *LeaveOneGroupOut* cross-validation scheme, where a group is defined as a single NBA season.
- Validation curves are visualized.
- Regression scores are calculated and listed.

### 6. Model training and predictions

- Models are fitted using training data, fitted models are used to predict test data.
- Feature importances for Decision Tree ensemble models (e.g. Random Forest) are calculated and listed.
- NBA player predictions are listed.

### 7. Final prediction

- Probability scores and corresponding rank on the scoring list are calculated for all models.
- The NBA MVP candidates are predicted according to each player's median scoring rank over all models.

## NBA MVP prediction 2018

The Top-5 NBA MVP candidates for 2018, ordered according to the mean scoring rank of all models:

1. ***James Harden*** 
2. ***LeBron James*** 
3. ***huh***
4. ***huh***
5. ***huh***