# Predicting the NBA All-Stars and NBA MVP with Machine Learning

The goal of this analysis is to predict the **NBA All-Stars** and **NBA MVP** for a given year, by applying various machine learning algorithms on NBA player performance data and All-Star selections / MVP voting data from other years.

## Data

NBA data are taken from [basketball-reference.com](https://www.basketball-reference.com). Data from 2000-2018 are included in the **data** directory of this repository, data from other years can be obtained by using [*Basketball_Reference_scraper.py*](Basketball_Reference_scraper.py).

## Analysis

The analyses are based on the [Scikit-learn](http://scikit-learn.org) machine learning package for Python, and are described in more detail in the following:

- [NBA All-Stars analysis README](README_AllStars.md)
- [NBA MVP analysis README](README_MVP.md)

## Predictions for 2018

#### NBA All-Stars 2018, based on 2010-2018 data:

The NBA All-Stars for 2018, in order of the median predicted calibrated probability score over all models and compared to the actual NBA All-Star selection in 2018:

- Western Conference:

	- **Deserved All-Stars:** *James Harden, Russell Westbrook, Anthony Davis, Kevin Durant, Damian Lillard, LaMarcus Aldridge, Jimmy Butler, DeMarcus Cousins, Stephen Curry, Karl-Anthony Towns*
	- **Questionable All-Stars:** *Paul George, Klay Thompson, Draymond Green*
	- **Snubbed non-All-Stars:** *Chris Paul, Nikola Jokic*

- Eastern Conference:

	- **Deserved All-Stars:** *Giannis Antetokounmpo, LeBron James, Kyrie Irving, Victor Oladipo, DeMar DeRozan, Joel Embiid, Andre Drummond, Kyle Lowry, Kemba Walker, Bradley Beal*
	- **Questionable All-Stars:** *John Wall, Kevin Love, Kristaps Porzingis, Al Horford, Goran Dragic*
	- **Snubbed non-All-Stars:** *Ben Simmons, Blake Griffin*

#### NBA MVP 2018, based on 2000-2018 data:

At the time of writing the NBA MVP for 2018 has not been awarded yet. The NBA MVP candidate top-5 for 2018, in order of the median predicted MVP voting share rank over all models:

1. ***James Harden*** (Median predicted MVS rank = 1, median predicted MVS = 0.784) 
2. ***LeBron James*** (Median predicted MVS rank = 2, median predicted MVS = 0.512) 
3. ***Kevin Durant*** (Median predicted MVS rank = 3, median predicted MVS = 0.284) 
4. ***Anthony Davis*** (Median predicted MVS rank = 4.5, median predicted MVS = 0.238) 
5. ***Russell Westbrook*** (Median predicted MVS rank = 5.5, median predicted MVS = 0.206)
