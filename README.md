# Predicting the NBA All-Stars and NBA Awards with Machine Learning

The goal of this analysis is to predict the [NBA All-Stars](https://www.basketball-reference.com/allstar/) and [NBA Awards](https://www.basketball-reference.com/awards/) for a specific year, by applying various machine learning algorithms on player performance data and All-Star selections / award voting data from other years. NBA Awards considered in this analysis are [Most Valuable Player](https://www.basketball-reference.com/awards/mvp.html) (MVP), [Rookie of the Year](https://www.basketball-reference.com/awards/roy.html) (ROY), [Defensive Player of the Year](https://www.basketball-reference.com/awards/dpoy.html) (DPOY) and [Sixth Man of the Year](https://www.basketball-reference.com/awards/smoy.html) (SMOY).

## Data

NBA data from 2000-2020 from [basketball-reference.com](https://www.basketball-reference.com) have been saved as csv-files in the [data](data) directory using the scraper functions in [*NBAanalysissetup.py*](NBAanalysissetup.py). Data from other years can be obtained by using [*Basketball_Reference_scraper.py*](Basketball_Reference_scraper.py). The following data are used in the analysis, where each file corresponds to a single season:

- *NBA_totals_[season].csv*: Regular player performance statistics
- *NBA_advanced_[season].csv*: Advanced player performance statistics
- *NBA_teammisc_[season].csv*: Team performance statistics
- *NBA_rookies_[season].csv*: Rookie performance statistics
- *NBA_allstars_[season].csv*: All-Star game statistics
- *NBA_MVP_[season].csv*: MVP voting results
- *NBA_ROY_[season].csv*: ROY voting results
- *NBA_DPOY_[season].csv*: DPOY voting results
- *NBA_SMOY_[season].csv*: SMOY voting results

## Analysis

The analysis is based on the [NumPy](http://www.numpy.org) and [Pandas](https://pandas.pydata.org) data analysis packages and the [Scikit-learn](http://scikit-learn.org) machine learning package for Python. The [XGBoost](http://xgboost.readthedocs.io/en/latest/) algorithm and the [Keras](https://keras.io/)-[TensorFlow](https://www.tensorflow.org/) deep learning libraries are tested as well. [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/) and [Bokeh](https://bokeh.pydata.org/) are used for visualization.

The NBA All-Stars and NBA Awards analyses are described in more detail in the following:

- [NBA All-Stars analysis README](README_AllStars.md)
- [NBA Awards analysis README](README_PlayerAwards.md)

## Predictions for 2020

At the time of writing (October 20, 2020), the NBA All-Stars for 2020 have been selected and the NBA Awards for 2020 have been awarded. Predictions for the NBA All-Stars and NBA Awards for 2020 are listed below.

### NBA All-Stars 2020:

The predicted NBA All-Stars for 2020 based on 2010-2020 data, ordered by the median predicted calibrated probability score over all selected models and compared to the actual NBA All-Star selection in 2020:

- Western Conference:

	- **Deserved All-Stars:** *Kawhi Leonard (LAC), Anthony Davis (LAL), James Harden (HOU), LeBron James (LAL), Luka Dončić (DAL), Russell Westbrook (HOU), Damian Lillard (POR), Devin Booker (PHO), Donovan Mitchell (UTA), Brandon Ingram (NOP)*
	- **Questionable All-Stars:** *Nikola Jokić (DEN), Chris Paul (OKC), Rudy Gobert (UTA)*
	- **Snubbed non-All-Stars:** *Paul George (LAC), Karl-Anthony Towns (MIN)*

- Eastern Conference:

	- **Deserved All-Stars:** *Giannis Antetokounmpo (MIL), Joel Embiid (PHI), Jayson Tatum (BOS), Trae Young (ATL), Pascal Siakam (TOR), Jimmy Butler (MIA), Kyle Lowry (TOR), Kemba Walker (BOS), Bam Adebayo (MIA), Khris Middleton (MIL)*
	- **Questionable All-Stars:** *Ben Simmons (PHI), Domantas Sabonis (IND)*
	- **Snubbed non-All-Stars:** *Bradley Beal (WAS), Zach LaVine (CHI)*

### NBA Awards 2020:

The top-3 predicted NBA MVP, ROY, DPOY and SMOY candidates based on 2000-2020 data are listed below in order of the median predicted Award Voting Share rank over all selected models. The award winners are shown in bold face.

- #### NBA Most Valuable Player 2020:

	1. ***Giannis Antetokounmpo (MIL)***
	2. *James Harden (HOU)* 
	3. *LeBron James (LAL)* 

- #### NBA Rookie of the Year 2020:

	1. ***Ja Morant (MEM)***
	2. *RJ Barrett (NYK)*
	3. *Kendrick Nunn (MIA)*

- #### NBA Defensive Player of the Year 2020:

	1. *Rudy Gobert (UTA)*
	2. ***Giannis Antetokounmpo (MIL)***
	3. *Anthony Davis (LAL)*

- #### NBA Sixth Man of the Year 2020:

	1. *Lou Williams (LAC)*
	2. *Dennis Schroder (OKC)*
	3. ***Montrezl Harrell (LAC)***
