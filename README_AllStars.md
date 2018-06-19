# Predicting the NBA All-Stars with Machine Learning

The goal of this analysis is to predict the [NBA All-Stars](https://www.basketball-reference.com/allstar/) for a specific year by applying machine learning algorithms on player performance data and All-Star selection data from other years. The analysis is based on the [NumPy](http://www.numpy.org) and [Pandas](https://pandas.pydata.org) data analysis packages and the [Scikit-learn](http://scikit-learn.org) machine learning package for Python. The [XGBoost](http://xgboost.readthedocs.io/en/latest/) algorithm and the [Keras](https://keras.io/)-[TensorFlow](https://www.tensorflow.org/) deep learning libraries are tested as well. The [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/) and [Bokeh](https://bokeh.pydata.org/) packages are used for visualization. NBA data are taken from [basketball-reference.com](https://www.basketball-reference.com). Data from 2000-2018 have been saved as csv-files in the [data](data) directory using the scraper functions in [*NBAanalysissetup.py*](NBAanalysissetup.py), data from other years can be obtained by using [*Basketball_Reference_scraper.py*](Basketball_Reference_scraper.py).

## Analysis

The analysis is presented as a [Python Jupyter Notebook](NBA_All-Stars.ipynb), and can be viewed online using [Jupyter nbviewer](https://nbviewer.jupyter.org/github/gmalim/NBA_analysis/blob/master/NBA_All-Stars.ipynb) (which has improved display rendering capabilities compared to Github). The outline of the analysis is summarized in the following:

### 1. Import prerequisites

- [NumPy](http://www.numpy.org)
- [Pandas](https://pandas.pydata.org)
- [Scikit-learn](http://scikit-learn.org)
- [Keras](https://keras.io/)
- [TensorFlow](https://www.tensorflow.org/)
- [XGBoost](http://xgboost.readthedocs.io/en/latest/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Bokeh](https://bokeh.pydata.org/)

### 2. User input

- Choose the first and last year for which data has been scraped, and choose the year you want to predict. The years that are not selected are used for cross-validation and training of the ML algorithms.
- Choose the minimum number of games a player has to have started per season to be included in the analysis.

### 3. Data processing

- Data loading and preparation (feature selection, *NaN* handling, etc.):
	- Features included in this analysis: *TW/TOT, G/TOT, GS/G, MP/G, 2P/48, 2P%, 3P/48, 3P%, FT/48, FT%, USG%, ORB%, DRB%, AST%, TOV%, STL%, BLK%, PF/48*. (Definitions can be found [here](https://www.basketball-reference.com/about/glossary.html)).
	- Target statistic is the players' All-Star selection status (*AS*), a binary variable.
- Feature scaling as required by various ML algorithms.
- Visualization of distributions of all features for All-Stars and non-All-Stars.

### 4. Unsupervised Learning: PCA & Clustering

- Principal Component Analysis is used for dimensionality reduction.
- Clustering algorithms to identify NBA All-Stars from non-All-Stars as separate groups in the data are tested and visualized:
	- *K-Means Clustering*
	- *Gaussian Mixture Model*
	- *Spectral Clustering* 
- Clustering performance scores for all clustering algorithms are calculated.

### 5. Supervised Learning: Classification

- Selection of various popular ML classification algorithms:
	- *Nearest Neighbours Classifier*
	- *Logistic Regression Classifier*
	- *Linear Support Vector Machine Classifier*
	- *Stochastic Gradient Descent Classifier*
	- *Linear Discriminant Analysis Classifier*
	- *Passive Aggressive Classifier*
	- *Perceptron Classifier*
	- *Neural Network Classifier*
	- *Deep Neural Network Classifier (Keras-Tensorflow)*
	- *Gaussian Process Classifier*
	- *Gaussian Naive Bayes Classifier*
	- *Decision Tree Classifier*
	- *Bagged Decision Tree Classifier*
	- *Random Forest Classifier*
	- *Extra Randomized Trees Classifier*
	- *Adaptive Boosted (AdaBoost) Decision Tree Classifier*
	- *Gradient Boosted Decision Tree Classifier*
	- *Extreme Gradient Boosted (XGBoost) Decision Tree Classifier*
- Hyper-parameter selection and instantiation of all models.

### 6. Cross-validation 

- All classifiers are cross-validated by using training data and the *LeaveOneGroupOut* cross-validation scheme, where a group is defined as a single NBA season.
- Validation curves are calculated and visualized.
- Classification scores for all models are calculated and listed.
- ROC and PR curves for all models are calculated and visualized.
- Feature importances for Decision Tree ensemble models are calculated and listed for all CV groups.
- Feature coefficients for linear models are calculated and listed for all CV groups.

### 7. Model training and predictions

- Models are fitted using training data, fitted models are used to predict test data.
- Confusion Matrices and classification scores for all models are calculated and visualized (if NBA All-Stars have been selected for test year).
- Feature importances for Decision Tree ensemble models are calculated and listed.
- Feature coefficients for linear models are calculated and listed.
- For the Logistic Regression Classifier, the fitted logistic curves corresponding to all data features are visualized.
- Decision function values / classification probability scores in 2-D feature space for all models are visualized.
- NBA player predictions for all models are listed.

### 8. Ensemble model

- An ensemble model of all selected classifiers with majority voting is created using the *VotingClassifier* class.
- Ensemble model is cross-validated and classification scores are calculated and listed.
- Ensemble model is fitted using training data, fitted ensemble is used to predict test data.
- NBA player predictions and classification results per classifier in the ensemble model are listed.

### 9. Probability scores and final prediction

- Classification probability scores and corresponding scoring ranks of test data are calculated for all models.
- Predicted score distributions and probability calibration curves of training data are visualized for all models.
- Calibrated classification probability scores and corresponding scoring ranks of test data are calculated for all models.
- Predicted calibrated score distributions and calibrated probability calibration curves of test data are vizualized for all models.
- The NBA All-Stars per conference are listed in order of the median predicted calibrated score over all selected models.

## NBA All-Star prediction 2018

At the time of writing the NBA All-Stars for 2018 were already selected. Therefore the analysis identifies three groups of players per conference:

1. **Deserved All-Stars:**     Players that were selected and are predicted as All-Stars.
2. **Questionable All-Stars:** Players that were selected but are not predicted as All-Stars.
3. **Snubbed non-All-Stars:**  Players that are predicted but were not selected as All-Stars.

The NBA players in these groups for 2018 are listed below, based on 2010-2018 data and in order of the median predicted calibrated scoring rank over all selected models:

- Western Conference:

	- **Deserved All-Stars:** *James Harden (HOU), Anthony Davis (NOP), Russell Westbrook (OKC), Kevin Durant (GSW), Damian Lillard (POR), DeMarcus Cousins (NOP), Jimmy Butler (MIN), Stephen Curry (GSW), LaMarcus Aldridge (SAS), Karl-Anthony Towns (MIN)*
	- **Questionable All-Stars:** *Klay Thompson (GSW), Paul George (OKC), Draymond Green (GSW)*
	- **Snubbed non-All-Stars:** *Chris Paul (HOU), Nikola Jokic (DEN)*

- Eastern Conference:

	- **Deserved All-Stars:** *LeBron James (CLE), Giannis Antetokounmpo (MIL), DeMar DeRozan (TOR), Joel Embiid (PHI), Kyrie Irving (BOS), Victor Oladipo (IND), Kevin Love (CLE), John Wall (WAS), Bradley Beal (WAS)*
	- **Questionable All-Stars:** *Kemba Walker (CHO), Andre Drummond (DET), Kyle Lowry (TOR), Kristaps Porzingis (NYK), Al Horford (BOS), Goran Dragic (MIA)*
	- **Snubbed non-All-Stars:** *Blake Griffin (DET), Ben Simmons (PHI), Dwight Howard (CHO)*

## Discussion

There are several caveats to the analysis:

- NBA player data used in the analysis are summary statistics over a complete season, while the NBA All-Star game is played halfway through a season. Therefore the analysis might actually be more suited to predict the selection of the All-NBA teams awarded at the end of the season.
- All-Star level players can be injured before the All-Star game but recover before the season ends and still pass the minimum number of games requirement to be included in the analysis.
- Similarly, players selected for the All-Star game who get injured before the All-Star game are replaced by other players who otherwise would not have been selected. (*In 2018 for instance, Paul George and Goran Dragic were injury replacements*). Therefore the injured players have been added to the scraped 2000-2018 csv-files by hand.
- All-Star level players can transfer between conferences during a season. (*In 2018 for instance, Blake Griffin transferred from the Clippers in the Western Conference to the Pistons in the Eastern Conference*).
- All-Star selection is not only determined by a player's individual performance, but also by his team's performance before the All-Star break. Team performance is included in the analysis by the *TW* statistic (i.e. the number of team wins per season), but no attempt has been made to tune the weight of this statistic compared to other data features. (*In 2018 for instance, Klay Thompson and  Draymond Green played for the Golden State Warriors, the defending NBA champions*).
- Similarly, All-Star selection is (partly) based on fan voting, and therefore popular players can get selected even if they played poorly during the season (*e.g. Kobe Bryant in 2014, 2015 and 2016*).
- The 1998–99 and 2011-2012 NBA seasons were shortened to 50 and 66 regular season games per team respectively due to a lock-out. Therefore there was no All-Star game in 1998-99, while the analysis might be suboptimal for 2011-2012.