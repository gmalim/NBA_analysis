# Predicting the NBA All-Stars and NBA Player Awards with Machine Learning

The goal of this analysis is to predict the [NBA All-Stars](https://www.basketball-reference.com/allstar/) and NBA Player Awards for a specific year, by applying various machine learning algorithms on player performance data and All-Star selections / award voting data from other years. Player Awards considered in this analysis are [Most Valuable Player](https://www.basketball-reference.com/awards/mvp.html) (MVP), [Rookie of the Year](https://www.basketball-reference.com/awards/roy.html) (ROY), [Defensive Player of the Year](https://www.basketball-reference.com/awards/dpoy.html) (DPOY) and [Sixth Man of the Year](https://www.basketball-reference.com/awards/smoy.html) (SMOY).

## Data

NBA data from 2000-2018 from [basketball-reference.com](https://www.basketball-reference.com) have been saved as csv-files in the [data](data) directory using the scraper functions in [*NBAanalysissetup.py*](NBAanalysissetup.py). Data from other years can be obtained by using [*Basketball_Reference_scraper.py*](Basketball_Reference_scraper.py). The following data are used in the analysis, where each file corresponds to a single season:

- *NBA_totals_[season].csv*: Regular player performance statistics
- *NBA_advanced_[season].csv*: Advanced player performance statistics
- *NBA_teammisc_[season].csv*: Team performance statistics
- *NBA_rookies_[season].csv*: Rookie performance statistics
- *NBA_allstars_[season].csv*: All-Star game statistics
- *NBA_MVP_[season].csv*: MVP voting statistics
- *NBA_ROY_[season].csv*: ROY voting statistics
- *NBA_DPOY_[season].csv*: DPOY voting statistics
- *NBA_SMOY_[season].csv*: SMOY voting statistics

## Analysis

The analysis is based on the [NumPy](http://www.numpy.org) and [Pandas](https://pandas.pydata.org) data analysis packages and the [Scikit-learn](http://scikit-learn.org) machine learning package for Python. The [XGBoost](http://xgboost.readthedocs.io/en/latest/) algorithm and the [Keras](https://keras.io/)-[TensorFlow](https://www.tensorflow.org/) deep learning libraries are tested as well. The [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/) and [Bokeh](https://bokeh.pydata.org/) packages are used for visualization.

The NBA All-Stars and NBA Player Awards analyses are described in more detail in the following:

- [NBA All-Stars analysis README](README_AllStars.md)
- [NBA Player Awards analysis README](README_PlayerAwards.md)

## Predictions for 2018

At the time of writing, the NBA All-Stars for 2018 have been selected, but the NBA Player Awards for 2018 have not been awarded yet. Predictions for the NBA All-Stars and Player Awards for 2018 are listed below.

### NBA All-Stars 2018:

The predicted NBA All-Stars for 2018 based on 2010-2018 data, ordered by the median predicted calibrated probability score over all selected models and compared to the actual NBA All-Star selection in 2018:

- Western Conference:

	- **Deserved All-Stars:** *James Harden (HOU), Anthony Davis (NOP), Russell Westbrook (OKC), Kevin Durant (GSW), Damian Lillard (POR), DeMarcus Cousins (NOP), Jimmy Butler (MIN), Stephen Curry (GSW), LaMarcus Aldridge (SAS), Karl-Anthony Towns (MIN)*
	- **Questionable All-Stars:** *Klay Thompson (GSW), Paul George (OKC), Draymond Green (GSW)*
	- **Snubbed non-All-Stars:** *Chris Paul (HOU), Nikola Jokic (DEN)*

- Eastern Conference:

	- **Deserved All-Stars:** *LeBron James (CLE), Giannis Antetokounmpo (MIL), DeMar DeRozan (TOR), Joel Embiid (PHI), Kyrie Irving (BOS), Victor Oladipo (IND), Kevin Love (CLE), John Wall (WAS), Bradley Beal (WAS)*
	- **Questionable All-Stars:** *Kemba Walker (CHO), Andre Drummond (DET), Kyle Lowry (TOR), Kristaps Porzingis (NYK), Al Horford (BOS), Goran Dragic (MIA)*
	- **Snubbed non-All-Stars:** *Blake Griffin (DET), Ben Simmons (PHI), Dwight Howard (CHO)*

### NBA Player Awards 2018:

The top-3 predicted NBA MVP, ROY, DPOY and SMOY candidates based on 2000-2018 data are listed below in order of the median predicted Award Voting Share rank over all selected models.

- #### NBA Most Valuable Player 2018:

	1. *James Harden (HOU)*
	2. *LeBron James (CLE)* 
	3. *Stephen Curry (GSW)* 

- #### NBA Rookie of the Year 2018:

	1. *Ben Simmons (PHI)*
	2. *Donovan Mitchell (UTA)*
	3. *Dennis Smith (DAL)*

- #### NBA Defensive Player of the Year 2018:

	1. *Draymond Green (GSW)*
	2. *Anthony Davis (NOP)*
	3. *Clint Capela (HOU)*

- #### NBA Sixth Man of the Year 2018:

	1. *Eric Gordon (HOU)*
	2. *Lou Williams (LAC)*
	3. *Marcus Smart (BOS)*
