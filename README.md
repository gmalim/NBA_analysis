# Predicting the NBA All-Stars with Machine Learning

The goal of this analysis is to predict the NBA All-Stars for a given year, based on NBA player data and All-Star selections in other years. This is accomplished by applying several machine learning classification algorithms on NBA player performance data. The analysis is based on the [Scikit-learn](http://scikit-learn.org) machine learning package for Python. NBA player data are taken from [basketball-reference.com](https://www.basketball-reference.com), data from 2000-2018 is included in the **data** directory of this repository. Data from other years can be obtained by using [*Basketball_Reference_scraper.py*](Basketball_Reference_scraper.py).  

## Analysis

The analysis is presented as a [Jupyter Notebook](NBA_All-Stars.ipynb). The outline of the analysis is summarized in the following:

### 1. Import external modules and libraries

- [NumPy](http://www.numpy.org)
- [Pandas](https://pandas.pydata.org)
- [Scikit-learn](http://scikit-learn.org)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)

### 2. User input

- Choose the year you want to predict, between 2000 and 2018. The years that are not selected are used for cross-validation and training of the ML algorithms.
- Choose whether you want to include advanced player statistics (e.g. *PER*, *VORP*, etc.) in the analysis or not.

### 3. NBA player data

- Data loading: NBA player data from 2000-2018 from [basketball-reference.com](https://www.basketball-reference.com) have been saved as csv-files in the **data** directory using the scraper functions in [*NBAanalysissetup.py*](NBAanalysissetup.py).
- Data preparation (feature selection, NaN handling, etc.).
- Feature scaling as required by various ML algorithms.

### 4. Unsupervised Learning

- Principal Component Analysis is used for dimensionality reduction.
- Clustering algorithms are tested to distinguish NBA All-Stars from non-All-Stars as separate groups in the data.

### 5. Supervised Learning

- Selection of various popular ML classification algorithms.
- Hyper-parameters tuning and instantiation of all models.

### 6. Cross-validation 

- All classifiers are cross-validated by using training data and the *LeaveOneGroupOut* cross-validation scheme, where a group is defined as a single NBA season.
- Classification scores are calculated. 
- ROC and PR curves are calculated and visualized.

### 7. Model training and predictions

- Models are fitted using training data, fitted models are used to predict test data.
- Confusion Matrices and classification scores are calculated.
- Feature importances and model coefficients are calculated.
- For the Logistic Regression Classifier, the fitted Logistic Curves corresponding to all data features are visualized.
- Decision function values / probability scores in 2-D feature space are visualized.
- NBA player predictions and are listed for each model.

### 8. Ensemble model

- An ensemble model with majority voting is created from the ML classifier list using the *VotingClassifier* class.
- Ensemble model is cross-validated and classification scores are calculated.
- Ensemble model is fitted using training data, fitted ensemble is used to predict test data.
- NBA player predictions and classification results per classifier in the ensemble model are listed.

### 9. Final prediction

- Prediction scores and corresponding placement on the scoring list are calculated for all models.
- The Western and Eastern Conference All-Stars are predicted according to a player's mean scoring placement averaged over all models.

## NBA All-Star prediction 2018

The analysis identifies three groups of NBA players per conference:

1. **Deserved NBA All-Stars:**     Players that were selected and predicted as All-Stars
2. **Questionable NBA All-Stars:** Players that were selected but not predicted as All-Stars
3. **Snubbed NBA non-All-Stars:**  Players that are predicted but not selected as All-Stars

For 2018, the NBA players in these groups are, in order of score:

- **Western Conference:**
	1. **Deserved NBA All-Stars:** James Harden, Russell Westbrook, Anthony Davis, Kevin Durant, Damian Lillard, Jimmy Butler, LaMarcus Aldridge, Karl-Anthony Towns, Stephen Curry, DeMarcus Cousins
	2. **Questionable NBA All-Stars:** Paul George, Draymond Green, Klay Thompson 
	3. **Snubbed NBA non-All-Stars:** Nikola Jokic, Chris Paul
- **Eastern Conference:**
	1. **Deserved NBA All-Stars:** LeBron James, Giannis Antetokounmpo, DeMar DeRozan, Joel Embiid, Kyrie Irving, Victor Oladipo, Andre Drummond, Kemba Walker, Bradley Beal
	2. **Questionable NBA All-Stars:** Kyle Lowry, Kevin Love, Kristaps Porzingis, John Wall, Al Horford, Goran Dragic
	3. **Snubbed NBA non-All-Stars:** Blake Griffin, Ben Simmons, Dwight Howard

