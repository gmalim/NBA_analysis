# Predicting the NBA All-Stars and NBA MVP with Machine Learning

The goal of this analysis is to predict the **NBA All-Stars** and **NBA MVP** for a given year, by applying various machine learning algorithms on NBA player performance data and All-Star selections / MVP voting data from other years.

## Data

NBA data are taken from [basketball-reference.com](https://www.basketball-reference.com). Data from 2000-2018 are included in the **data** directory of this repository, data from other years can be obtained by using [*Basketball_Reference_scraper.py*](Basketball_Reference_scraper.py).

## Analysis

The analyses are based on the [Scikit-learn](http://scikit-learn.org) machine learning package for Python, and are described in more detail in the following:

- [NBA All-Stars analysis README](README_AllStars.md)
- [NBA MVP analysis README](README_MVP.md)

## Predictions for 2018

#### NBA All-Stars (based on 2010-2018 data):

The NBA All-Stars for 2018, ordered according to the median predicted probability score over all models and compared to the actual NBA All-Star selection in 2018:

- Western Conference:

	- **Deserved NBA All-Stars:** *James Harden, Russell Westbrook, Kevin Durant, Damian Lillard, Anthony Davis, LaMarcus Aldridge, Stephen Curry, DeMarcus Cousins, Jimmy Butler, Karl-Anthony Towns*
	- **Questionable NBA All-Stars:** *Paul George, Klay Thompson, Draymond Green*
	- **Snubbed NBA non-All-Stars:** *Chris Paul, Nikola Jokic*

- Eastern Conference:

	- **Deserved NBA All-Stars:** *LeBron James, Giannis Antetokounmpo, DeMar DeRozan, Victor Oladipo, Kyrie Irving, Joel Embiid, Kyle Lowry, Andre Drummond, Kemba Walker, John Wall*
	- **Questionable NBA All-Stars:** *Bradley Beal, Kevin Love, Kristaps Porzingis, Al Horford, Goran Dragic*
	- **Snubbed NBA non-All-Stars:** *Blake Griffin, Ben Simmons*

#### NBA MVP (based on 2000-2018 data):

The NBA MVP candidate top-5 for 2018 according to the median predicted probability scoring-rank of all models (the actual NBA MVP 2018 is not known yet):

1. ***James Harden*** (Median scoring-rank = 1, median score = 0.784) 
2. ***LeBron James*** (Median scoring-rank = 2, median score = 0.512) 
3. ***Kevin Durant*** (Median scoring-rank = 3, median score = 0.284) 
4. ***Anthony Davis*** (Median scoring-rank = 4.5, median score = 0.238) 
5. ***Russell Westbrook*** (Median scoring-rank = 5.5, median score = 0.206)
