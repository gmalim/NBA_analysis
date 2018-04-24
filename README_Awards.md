# Predicting the NBA MVP with Machine Learning

The goal of this analysis is to predict the NBA MVP for a given year, based on NBA player data and MVP voting statistics in other years. This is accomplished by applying several machine learning regression algorithms on NBA player performance data. The analysis is based on the [Scikit-learn](http://scikit-learn.org) machine learning package for Python. NBA data are taken from [basketball-reference.com](https://www.basketball-reference.com). Data from 2000-2018 is included in the **data** directory of this repository, data from other years can be obtained by using [*Basketball_Reference_scraper.py*](Basketball_Reference_scraper.py).  

## Analysis

The analysis is presented as a [Jupyter Notebook](NBA_MVP.ipynb), and can be viewed online using [Jupyter nbviewer](https://nbviewer.jupyter.org/github/gmalim/NBA_analysis/blob/master/NBA_MVP.ipynb) (which has improved display rendering capabilities compared to Github). The outline of the analysis is summarized in the following:

### 1. Import external modules and libraries

- [NumPy](http://www.numpy.org)
- [Pandas](https://pandas.pydata.org)
- [Scikit-learn](http://scikit-learn.org)
- [XGBoost](http://xgboost.readthedocs.io/en/latest/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)

### 2. User input

- Choose the year you want to predict, between 2000 and 2018. The years that are not selected are used for cross-validation and training of the ML algorithms.
- Choose whether you want to include advanced player statistics (e.g. *PER*, *VORP*, etc.) in the analysis or not.
- Choose the minimum number of games a player has to have started per season to be included in the analysis.

### 3. Data handling

- Data loading: NBA data from 2000-2018 from [basketball-reference.com](https://www.basketball-reference.com) have been saved as csv-files in the **data** directory using the scraper functions in [*NBAanalysissetup.py*](NBAanalysissetup.py).
- Data preparation (feature selection, *NaN* handling, etc.).
- Features included in this analysis: *G, GS, MP/G, PTS/48, PER, TS%, TRB%, AST%, STL%, BLK%, USG%, OWS/48, DWS/48, OBPM, DBPM, VORP, TW*. (Definitions can be found [here](https://www.basketball-reference.com/about/glossary.html)).
- Target statistic is the players' MVP voting share (*MVS*), a continuous variable.
- Relationships between MVS and other features in training data are visualized.
- Feature scaling as required by various ML algorithms.

### 4. Supervised Learning

- Selection of various popular ML regression algorithms:
	- *Nearest Neighbours Regressor*
	- *Ridge Regressor*
	- *Lasso Regressor*
	- *ElasticNet Regressor*
	- *Support Vector Machine Regressor*
	- *Stochastic Gradient Descent Regressor*
	- *Passive Aggressive Regressor*
	- *Neural Network Regressor*
	- *Gaussian Process Regressor*
	- *Decision Tree Regressor*
	- *Random Forest Regressor*
	- *Extra Randomized Trees Regressor*
	- *Adaptive Boosted (AdaBoost) Decision Tree Regressor*
	- *Gradient Boosted Decision Tree Regressor*
	- *Extreme Gradient Boosted (XGBoost) Decision Tree Regressor*
- Hyper-parameter selection and instantiation of all models.

### 5. Cross-validation 

- All regressors are cross-validated by using training data and the *LeaveOneGroupOut* cross-validation scheme, where a group is defined as a single NBA season.
- Validation curves are calculated and visualized.
- Regression metrics are calculated and listed for all models.
- Feature importances for Decision Tree ensemble models (e.g. Random Forest) are calculated and listed for all CV groups.

### 6. Model training and predictions

- Models are fitted using training data, fitted models are used to predict test data.
- Regression metrics are calculated and listed for all models if the NBA MVP has been awarded for test year.
- NBA player predictions for all models are listed.
- The NBA MVP candidates are listed in order of the median predicted MVS rank over all models.

## NBA MVP prediction 2018

At the time of writing the NBA MVP for 2018 has not been awarded yet. The MVP candidate top-5, based on 2000-2018 data, in order of the median predicted MVS rank over all models:

1. ***James Harden*** (Median predicted MVS rank = 1, median predicted MVS = 0.784) 
2. ***LeBron James*** (Median predicted MVS rank = 2, median predicted MVS = 0.512) 
3. ***Kevin Durant*** (Median predicted MVS rank = 3, median predicted MVS = 0.284) 
4. ***Anthony Davis*** (Median predicted MVS rank = 4.5, median predicted MVS = 0.238) 
5. ***Russell Westbrook*** (Median predicted MVS rank = 5.5, median predicted MVS = 0.206) 

## Discussion

There are several caveats to the analysis:

- The MVS absolute values predicted by the accepted models are not very accurate. This is due to the small sample size of players with MVP votes and the highly non-linear relationships between MVS and other features in the analysis. Therefore the predicted MVS rank instead of the absolute value is used as a measure for MVP candidateship.