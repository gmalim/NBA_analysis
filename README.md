# Predicting the NBA All-Stars and NBA Player Awards with Machine Learning

The goal of this analysis is to predict the **NBA All-Stars** and **NBA Player Awards** for a given year, by applying various machine learning algorithms on player performance data and All-Star selections / award voting data from other years. Player Awards considered in this analysis are **Most Valuable Player** (MVP), **Rookie of the Year** (ROY) and **Defensive Player of the Year** (DPOY).

## Data

NBA data are taken from [basketball-reference.com](https://www.basketball-reference.com). Data from 2000-2018 are included in the **data** directory of this repository, data from other years can be obtained by using [*Basketball_Reference_scraper.py*](Basketball_Reference_scraper.py).

## Analysis

The analyses are based on the [Scikit-learn](http://scikit-learn.org) machine learning package for Python, and are described in more detail in the following:

- [NBA All-Stars analysis README](README_AllStars.md)
- [NBA Player Awards analysis README](README_PlayerAwards.md)

## Predictions for 2018

At the time of writing, the NBA All-Stars for 2018 have been selected, but the NBA Player Awards for 2018 have not been awarded yet. Predictions for the NBA All-Stars and Player Awards for 2018 are listed below. The All-Stars analysis is based on 2010-2018 data, the Player Awards analysis is based on 2000-2018 data.

### NBA All-Stars 2018:

The NBA All-Stars for 2018, ordered by to the median predicted calibrated probability score over all selected models and compared to the actual NBA All-Star selection in 2018:

- Western Conference:

	- **Deserved All-Stars:** *James Harden (HOU), Russell Westbrook (OKC), Anthony Davis (NOP), Kevin Durant (GSW), Damian Lillard (POR), LaMarcus Aldridge (SAS), Jimmy Butler (MIN), DeMarcus Cousins (NOP), Stephen Curry (GSW), Karl-Anthony Towns (MIN)*
	- **Questionable All-Stars:** *Paul George (OKC), Klay Thompson (GSW), Draymond Green (GSW)*
	- **Snubbed non-All-Stars:** *Chris Paul (HOU), Nikola Jokic (DEN)*

- Eastern Conference:

	- **Deserved All-Stars:** *Giannis Antetokounmpo (MIL), LeBron James (CLE), Kyrie Irving (BOS), Victor Oladipo (IND), DeMar DeRozan (TOR), Joel Embiid (PHI), Andre Drummond (DET), Kyle Lowry (TOR), Kemba Walker (CHO), Bradley Beal (WAS)*
	- **Questionable All-Stars:** *John Wall (WAS), Kevin Love (CLE), Kristaps Porzingis (NYK), Al Horford (BOS), Goran Dragic (MIA)*
	- **Snubbed non-All-Stars:** *Ben Simmons (PHI), Blake Griffin (DET)*

### NBA Player Awards 2018:

The predicted NBA MVP, ROY and DPOY candidate Top-3s, based on 2000-2018 data, are listed below in order of the median predicted AVS rank over all selected models.

- #### NBA Most Valuable Player 2018:

	1. *James Harden (HOU)*
	2. *LeBron James (CLE)*
	3. *Kevin Durant (GSW)* 

- #### NBA Rookie of the Year 2018:

	1. *Ben Simmons (PHI)*
	2. *Donovan Mitchell (UTA)*
	3. *Lauri Markkanen (CHI)*

- #### NBA Defensive Player of the Year 2018:

	1. *Andre Drummond (DET)*
	2. *Clint Capela (HOU)*
	3. *Anthony Davis (NOP)*

