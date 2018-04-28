# Predicting NBA Player Award winners with Machine Learning

The goal of this analysis is to predict NBA Player Award winners for a specific year. This is accomplished by applying several machine learning regression algorithms on player performance data and award voting data in other years. Player Awards considered in this analysis are [Most Valuable Player](https://www.basketball-reference.com/awards/mvp.html) (MVP), [Rookie of the Year](https://www.basketball-reference.com/awards/roy.html) (ROY), [Defensive Player of the Year](https://www.basketball-reference.com/awards/dpoy.html) (DPOY) and [Sixth Man of the Year](https://www.basketball-reference.com/awards/smoy.html) (SMOY). The analysis is based on the [Scikit-learn](http://scikit-learn.org) machine learning package for Python. NBA data are taken from [basketball-reference.com](https://www.basketball-reference.com). Data from 2000-2018 have been saved as csv-files in the [data](data) directory using the scraper functions in [*NBAanalysissetup.py*](NBAanalysissetup.py), data from other years can be obtained by using [*Basketball_Reference_scraper.py*](Basketball_Reference_scraper.py).  

## Analysis

The analysis is presented as a [Python Jupyter Notebook](NBA_PlayerAwards.ipynb), and can be viewed online using [Jupyter nbviewer](https://nbviewer.jupyter.org/github/gmalim/NBA_analysis/blob/master/NBA_PlayerAwards.ipynb) (which has improved display rendering capabilities compared to Github). The outline of the analysis is summarized in the following:

### 1. Import external modules and libraries

- [NumPy](http://www.numpy.org)
- [Pandas](https://pandas.pydata.org)
- [Scikit-learn](http://scikit-learn.org)
- [XGBoost](http://xgboost.readthedocs.io/en/latest/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)

### 2. User input

- Choose the NBA Player Award you want to predict:
	- NBA Most Valuable Player (MVP)
	- NBA Rookie of the Year (ROY)
	- NBA Defensive Player of the Year (DPOY)
	- NBA Sixth Man of the Year (SMOY)
- Choose the first and last year for which data has been scraped, and choose the year you want to predict. The years that are not selected are used for cross-validation and training of the ML algorithms.
- Choose whether you want to include advanced player statistics (e.g. *PER*, *VORP*, etc.) in the analysis or not.
- Choose the minimum number of games a player has to have played (ROY) or started (MVP, DPOY) per season to be included in the analysis.

### 3. Data handling

- Data loading and preparation (feature selection, *NaN* handling, etc.).
	- Features included in this analysis: *G, GS, MP/G, PTS/48, PER, TS%, TRB%, AST%, STL%, BLK%, USG%, OWS/48, DWS/48, OBPM, DBPM, VORP, TW*. (Definitions can be found [here](https://www.basketball-reference.com/about/glossary.html)).
	- Target statistic is the players' Award Voting Share (*AVS*), a continuous variable.
- The AVS distributions of training and test data as well as the relationships between AVS and other features in training data are visualized.
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
- Predicted versus true AVS distributions are visualized for all models and regression metrics are calculated.
- Feature importances for Decision Tree ensemble models are calculated and listed for all CV groups.

### 6. Model training and predictions

- Models are fitted using training data, fitted models are used to predict test data.
- Predicted versus true AVS distributions are visualized for all models and regression metrics are calculated if the NBA Player Award has been awarded for test year.
- NBA player predictions for all models are listed.
- The NBA Player Award candidates are listed in order of the median predicted AVS rank over all selected models.

## NBA Player Awards predictions for 2018

At the time of writing the NBA Player Awards for 2018 have not been awarded yet. The top-3 predicted NBA MVP, ROY and DPOY candidates based on 2000-2018 data are listed below in order of the median predicted AVS rank over all selected models.

- ### NBA Most Valuable Player 2018:

	1. *James Harden (HOU)*
	2. *LeBron James (CLE)* 
	3. *Russell Westbrook (OKC)* 

- ### NBA Rookie of the Year 2018:

	1. *Ben Simmons (PHI)*
	2. *Donovan Mitchell (UTA)*
	3. *Kyle Kuzma (LAL)*

- ### NBA Defensive Player of the Year 2018:

	1. *Andre Drummond (DET)*
	2. *Clint Capela (HOU)*
	3. *Ben Simmons (PHI)*

- ### NBA Sixth Man of the Year 2018:

	1. *Lou Williams (LAC)*
	2. *Eric Gordon (HOU)*
	3. *Jordan Clarkson (CLE)*

## Discussion

Unfortunately none of the regression models tested in this analysis are able to accurately predict the Player Award voting shares. This is due to the erratic relationship between AVS and other features in the analysis (and to a lesser degree to the small sample size of players with award votes). Therefore the predicted absolute AVS values are ignored and the predicted median AVS rank over all models is used as a measure for Player Award candidateship.