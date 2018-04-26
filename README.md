# Predicting the NBA All-Stars and NBA Player Awards with Machine Learning

The goal of this analysis is to predict the **NBA All-Stars** and **NBA Player Awards** for a given year, by applying various machine learning algorithms on player performance data and All-Star selections / award voting data from other years. Player Awards considered in this analysis are **Most Valuable Player** (MVP), **Rookie of the Year** (ROY), **Defensive Player of the Year** (DPOY) and **Sixth Man of the Year** (SMOY).

## Data

NBA data from 2000-2018 from [basketball-reference.com](https://www.basketball-reference.com) have been saved as csv-files in the **data** directory using the scraper functions in [*NBAanalysissetup.py*](NBAanalysissetup.py). Data from other years can be obtained by using [*Basketball_Reference_scraper.py*](Basketball_Reference_scraper.py). The following data are used in the analysis, where each file corresponds to a single season:

- *NBA_totals_xxx.csv*: Regular player performance statistics
- *NBA_advanced_xxx.csv*: Advanced player performance statistics
- *NBA_teammisc_xxx.csv*: Team performance statistics
- *NBA_rookies_xxx.csv*: Rookie performance statistics
- *NBA_allstars_xxx.csv*: All-Star game statistics
- *NBA_MVP_xxx.csv*: MVP voting statistics
- *NBA_ROY_xxx.csv*: ROY voting statistics
- *NBA_DPOY_xxx.csv*: DPOY voting statistics
- *NBA_SMOY_xxx.csv*: SMOY voting statistics

## Analysis

The analyses are based on the [Scikit-learn](http://scikit-learn.org) machine learning package for Python, and are described in more detail in the following:

- [NBA All-Stars analysis README](README_AllStars.md)
- [NBA Player Awards analysis README](README_PlayerAwards.md)

## Predictions for 2018

At the time of writing, the NBA All-Stars for 2018 have been selected, but the NBA Player Awards for 2018 have not been awarded yet. Predictions for the NBA All-Stars and Player Awards for 2018 are listed below.

### NBA All-Stars 2018:

The NBA All-Stars for 2018, based on 2010-2018 data, ordered by to the median predicted calibrated probability score over all selected models and compared to the actual NBA All-Star selection in 2018:

- Western Conference:

	- **Deserved All-Stars:** *James Harden (HOU), Russell Westbrook (OKC), Anthony Davis (NOP), Kevin Durant (GSW), Damian Lillard (POR), LaMarcus Aldridge (SAS), Jimmy Butler (MIN), DeMarcus Cousins (NOP), Stephen Curry (GSW), Karl-Anthony Towns (MIN)*
	- **Questionable All-Stars:** *Paul George (OKC), Klay Thompson (GSW), Draymond Green (GSW)*
	- **Snubbed non-All-Stars:** *Chris Paul (HOU), Nikola Jokic (DEN)*

- Eastern Conference:

	- **Deserved All-Stars:** *Giannis Antetokounmpo (MIL), LeBron James (CLE), Kyrie Irving (BOS), Victor Oladipo (IND), DeMar DeRozan (TOR), Joel Embiid (PHI), Andre Drummond (DET), Kyle Lowry (TOR), Kemba Walker (CHO), Bradley Beal (WAS)*
	- **Questionable All-Stars:** *John Wall (WAS), Kevin Love (CLE), Kristaps Porzingis (NYK), Al Horford (BOS), Goran Dragic (MIA)*
	- **Snubbed non-All-Stars:** *Ben Simmons (PHI), Blake Griffin (DET)*

### NBA Player Awards 2018:

The top-3 predicted NBA MVP, ROY, DPOY and SMOY candidates based on 2000-2018 data are listed below in order of the median predicted AVS rank over all selected models.

- #### NBA Most Valuable Player 2018:

	1. *James Harden (HOU)*
	2. *LeBron James (CLE)* 
	3. *Russell Westbrook (OKC)* 

- #### NBA Rookie of the Year 2018:

	1. *Ben Simmons (PHI)*
	2. *Donovan Mitchell (UTA)*
	3. *Kyle Kuzma (LAL)*

- #### NBA Defensive Player of the Year 2018:

	1. *Andre Drummond (DET)*
	2. *Clint Capela (HOU)*
	3. *Ben Simmons (PHI)*

- #### NBA Sixth Man of the Year 2018:

	1. *Lou Williams (LAC)*
	2. *Eric Gordon (HOU)*
	3. *Jordan Clarkson (CLE)*

