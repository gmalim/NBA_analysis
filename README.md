# Predicting the NBA All-Stars and NBA MVP with Machine Learning

The goal of this analysis is to predict the **NBA All-Stars** and **NBA MVP** for a given year, by applying various machine learning algorithms on NBA player performance data and All-Star selections / MVP voting data from other years.

## Data

NBA data are taken from [basketball-reference.com](https://www.basketball-reference.com). Data from 2010-2018 are included in the **data** directory of this repository, data from other years can be obtained by using [*Basketball_Reference_scraper.py*](Basketball_Reference_scraper.py).

## Analysis

The analyses are based on the [Scikit-learn](http://scikit-learn.org) machine learning package for Python, and are described in more detail in the following:

- [NBA All-Stars analysis README](README_AllStars.md)
- [NBA MVP analysis README](README_MVP.md)

## Predictions for 2018

#### NBA All-Stars:

The NBA All-Stars for 2018, ordered according to the median scoring rank over all models:

- Western Conference:

	- **Deserved NBA All-Stars:** *James Harden, Anthony Davis, Russell Westbrook, Kevin Durant, Damian Lillard, LaMarcus Aldridge, Jimmy Butler, DeMarcus Cousins, Stephen Curry, Karl-Anthony Towns*
	- **Questionable NBA All-Stars:** *Paul George, Klay Thompson, Draymond Green*
	- **Snubbed NBA non-All-Stars:** *Chris Paul, Nikola Jokic*

- Eastern Conference:

	- **Deserved NBA All-Stars:** *LeBron James, Giannis Antetokounmpo, DeMar DeRozan, Kyrie Irving, Victor Oladipo, Joel Embiid, Kemba Walker, Andre Drummond, Kyle Lowry, Bradley Beal*
	- **Questionable NBA All-Stars:** *John Wall, Kevin Love, Al Horford, Kristaps Porzingis, Goran Dragic*
	- **Snubbed NBA non-All-Stars:** *Ben Simmons, Blake Griffin*

#### NBA MVP:

The Top-5 NBA MVP candidates for 2018, ordered according to the mean scoring rank over all models:

1. ***James Harden*** (Mean scoring rank = 1, median score = 0.744) 
2. ***LeBron James*** (Mean scoring rank = 2, median score = 0.468) 
3. ***Russell Westbrook*** (Mean scoring rank = 4.2, median score = 0.240) 
4. ***Kevin Durant*** (Mean scoring rank = 4.4, median score = 0.236) 
5. ***Anthony Davis*** (Mean scoring rank = 5.6, median score = 0.184) 

