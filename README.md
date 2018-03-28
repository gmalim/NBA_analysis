# Predicting the NBA All-Stars with Machine Learning

The goal of this analysis is to predict the NBA All-Stars for a given year, based on NBA player data and All-Star selections in other years. This is accomplished by applying several machine learning classification algorithms on player performance statistics per season. The analysis is based on the [Scikit-learn](http://scikit-learn.org) machine learning package, NBA player data are taken from [basketball-reference.com](https://www.basketball-reference.com). 

## NBA player data:

NBA player data from 2000-2018 from [basketball-reference.com](https://www.basketball-reference.com) have been saved as csv-files in the **data** directory using the scraper functions in [*NBAanalysissetup.py*](NBAanalysissetup.py). You can use [*Basketball_Reference_scraper.py*](Basketball_Reference_scraper.py) to scrape new data.

Statistics taken into account in the analysis:

| Stat     | Acronym |
|----------|---------|
| *G*      | Games played  |
| *GS*     | Games started |
| *MP/G*   | Minutes played per game |
| *2P/48*  | 2P Field goals per 48 minutes |
| *2PA/48* | 2P Field goal attempts per 48 minutes |
| *3P/48*  | 3P Field goals per 48 minutes |
| *3PA/48* | 3P Field goal attempts per 48 minutes |
| *FT/48*  | Free throws per 48 minutes |
| *FTA/48* | Free throw attempts per 48 minutes |
| *ORB/48* | Offensive rebounds per 48 minutes |
| *DRB/48* | Defensive rebounds per 48 minutes |
| *AST/48* | Assists per 48 minutes |
| *TOV/48* | Turnovers per 48 minutes |
| *STL/48* | Steals per 48 minutes |
| *BLK/48* | Blocks per 48 minutes |
| *PF/48*  | Personal fouls per 48 minutes |
| *PTS/48* | Points per 48 minutes |

Advanced statistics that can be included in the analysis:

| Stat  | Acronym | Definition |
|-------|---------|-------------|
| *PER* | Player Efficiency Rating | A per-minute rating of a player's overall performance with respect to an average player, see [here](https://www.basketball-reference.com/about/per.html)
| *ORB%* | Offensive Rebound Percentage | Estimate of the percentage of available offensive rebounds a player grabbed while he was on the floor: *ORB%* = 100 * (*ORB* * (*Tm MP* / 5)) / (*MP* * (*Tm ORB* + *Opp DRB*)) |
| *DRB%* | Defensive Rebound Percentage | Estimate of the percentage of available defensive rebounds a player grabbed while he was on the floor: *ODRB%* = 100 * (*DRB* * (*Tm MP* / 5)) / (*MP* * (*Tm DRB* + *Opp ORB*)) |
| *AST%* | Assist Percentage | Estimate of the percentage of teammate field goals a player assisted while he was on the floor: *AST%* = 100 * *AST* / (((*MP* / (*Tm MP* / 5)) * *Tm FG*) - *FG*) |
| *STL%* | Steal Percentage | Estimate of the percentage of opponent possessions that end with a steal by the player while he was on the floor: *STL%* = 100 * (*STL* * (*Tm MP* / 5)) / (*MP* * *Opp Poss*) |
| *BLK%* | Block Percentage | Estimate of the percentage of opponent 2P field goal attempts blocked by the player while he was on the floor: *BLK%* = 100 * (*BLK* * (*Tm MP* / 5)) / (*MP* * (*Opp FGA* - *Opp 3PA*)) |
| *TOV%* | Turnover Percentage | Estimate of turnovers per 100 plays: *TOV%* = 100 * *TOV* / (*FGA* + 0.44 * *FTA* + *TOV*) |
| *USG%* | Usage Percentage | Estimate of the percentage of team plays used by a player while he was on the floor: *USG%* = 100 * ((*FGA* + 0.44 * *FTA* + *TOV*) * (*Tm MP* / 5)) / (*MP* * (*Tm FGA* + 0.44 * *Tm FTA* + *Tm TOV*)) |
| *OWS/48* | Offensive Win Shares per 48 mins | See [here](https://www.basketball-reference.com/about/ws.html)
| *DWS/48* | Defensive Win Shares per 48 mins | See [here](https://www.basketball-reference.com/about/ws.html)
| *OBPM* | Offensive Box Plus/Minus | OBPM is a per-100-possession stat, see [here](https://www.basketball-reference.com/about/bpm.html)
| *DBPM* | Defensive Box Plus/Minus | DBPM is a per-100-possession stat, see [here](https://www.basketball-reference.com/about/bpm.html)
| *VORP* | Value Over Replacement Player | VORP is a per-100-possession stat, see [here](https://www.basketball-reference.com/about/bpm.html) |

## ML classifier algorithms that are implemented in the analysis:

1. Logistic Regression Classifier
2. Gaussian Naive Bayes Classifier
3. Gaussian Process Classifier
4. Nearest Neighbours Classifier
5. Linear Support Vector Machine Classifier
6. Decision Tree Classifier
7. Random Forest Classifier
8. Extra Trees Classifier
9. Gradient Tree Boosting Classifier
10. Ada Boost Classifier
11. Quadratic Discriminant Analysis Classifier
12. Neural Network Classifier

## Analysis:

The analysis is implemented as a python program ([*NBA_All-Stars.py*](NBA_All-Stars.py)) and, equivalently, as a Jupyter notebook ([*NBA_All-Stars.ipynb*](NBA_All-Stars.ipynb)). Helper functions are defined in [*NBAanalysissetup.py*](NBAanalysissetup.py). 

To run the analysis:

- Choose the year you want to use to run some validation tests (*validation_year*) and the year you want to predict (*predicion_year*), both in range 2000-2018. The years that are not selected for validation tests and prediction are used to train the model.
- Choose whether you want to include advanced player statistics (e.g. *PER*, *WS*, etc.) in the model or not.
- Choose a ML classifier from the list above.
