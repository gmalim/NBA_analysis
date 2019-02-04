# Predicting the NBA All-Stars and NBA Awards with Machine Learning

The goal of this analysis is to predict the [NBA All-Stars](https://www.basketball-reference.com/allstar/) and [NBA Awards](https://www.basketball-reference.com/awards/) for a specific year, by applying various machine learning algorithms on player performance data and All-Star selections / award voting data from other years. NBA Awards considered in this analysis are [Most Valuable Player](https://www.basketball-reference.com/awards/mvp.html) (MVP), [Rookie of the Year](https://www.basketball-reference.com/awards/roy.html) (ROY), [Defensive Player of the Year](https://www.basketball-reference.com/awards/dpoy.html) (DPOY) and [Sixth Man of the Year](https://www.basketball-reference.com/awards/smoy.html) (SMOY).

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

The NBA All-Stars and NBA Awards analyses are described in more detail in the following:

- [NBA All-Stars analysis README](README_AllStars.md)
- [NBA Awards analysis README](README_PlayerAwards.md)

## Predictions for 2019

At the time of writing (February 4, 2019), the NBA All-Stars for 2019 have been selected but the NBA Awards for 2019 have not been awarded yet. Predictions for the NBA All-Stars and NBA Awards for 2019 are listed below.

### NBA All-Stars 2019:

The predicted NBA All-Stars for 2019 based on 2010-2019 data, ordered by the median predicted calibrated probability score over all selected models and compared to the actual NBA All-Star selection in 2019:

- Western Conference:

	- **Deserved All-Stars:** *James Harden (HOU), Kevin Durant (GSW), Anthony Davis (NOP), Paul George (OKC), Stephen Curry (GSW), Damian Lillard (POR), LeBron James (LAL), Russell Westbrook (OKC), LaMarcus Aldridge (SAS), Nikola Jokic (DEN), Karl-Anthony Towns (MIN)*
	- **Questionable All-Stars:** *Klay Thompson (GSW)*
	- **Snubbed non-All-Stars:** *DeMar DeRozan (SAS)*

- Eastern Conference:

	- **Deserved All-Stars:** *Joel Embiid (PHI), Giannis Antetokounmpo (MIL), Kawhi Leonard (TOR), Blake Griffin (DET), Kemba Walker (CHO), Kyrie Irving (BOS), Nikola Vucevic (ORL), Ben Simmons (PHI), D'Angelo Russell (BRK), Bradley Beal (WAS), Khris Middleton (MIL)*
	- **Questionable All-Stars:** *Kyle Lowry (TOR), Victor Oladipo (IND)*
	- **Snubbed non-All-Stars:** *Jimmy Butler (PHI), Eric Bledsoe (MIL)*

### NBA Awards 2019 (at the time of writing: February 4, 2019):

The top-3 predicted NBA MVP, ROY, DPOY and SMOY candidates based on 2000-2019 data are listed below in order of the median predicted Award Voting Share rank over all selected models.

- #### NBA Most Valuable Player 2019:

	1. *James Harden (HOU)*
	2. *Kevin Durant (GSW)* 
	3. *Joel Embiid (PHI)* 

- #### NBA Rookie of the Year 2019:

	1. *Luka Doncic (DAL)*
	2. *Trae Young (ATL)*
	3. *Deandre Ayton (PHO)*

- #### NBA Defensive Player of the Year 2019:

	1. *Kawhi Leonard (TOR)*
	2. *Joel Embiid (PHI)*
	3. *Myles Turner (IND)*

- #### NBA Sixth Man of the Year 2019:

	1. *Lou Williams (LAC)*
	2. *Spencer Dinwiddie (BRK)*
	3. *Dennis Schroder (OKC)*
