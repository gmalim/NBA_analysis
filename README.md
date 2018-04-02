# Predicting the NBA All-Stars with Machine Learning

The goal of this analysis is to predict the NBA All-Stars for a given year, based on NBA player data and All-Star selections in other years. This is accomplished by applying several machine learning classification algorithms on NBA player performance data. The analysis is based on the [Scikit-learn](http://scikit-learn.org) machine learning package, NBA player data are taken from [basketball-reference.com](https://www.basketball-reference.com). 

## NBA player data

NBA player data from 2000-2018 from [basketball-reference.com](https://www.basketball-reference.com) have been saved as csv-files in the **data** directory using the scraper functions in [*NBAanalysissetup.py*](NBAanalysissetup.py). New data can be scraped by using [*Basketball_Reference_scraper.py*](Basketball_Reference_scraper.py). 

The following total statistics per season are taken into account in the analysis:

| Stat   | Acronym | Note |
|--------|---------|------|
| *G*    | Games played |
| *GS/G* | Games started / Games played |
| *MP/G* | Minutes per game |
| *3P*   | 3P field goals |
| *3PA*  | 3P field goal attempts |
| *2P*   | 2P field goals |
| *2PA*  | 2P field goal attempts |
| *FT*   | Free throws |
| *FTA*  | Free throw attempts |
| *ORB*  | Offensive rebounds | Replaced by *ORB%* if advanced stats are included |
| *DRB*  | Defensive rebounds | Replaced by *DRB%* if advanced stats are included |
| *AST*  | Assists | Replaced by *AST%* if advanced stats are included |
| *STL*  | Steals | Replaced by *STL%* if advanced stats are included |
| *BLK*  | Blocks | Replaced by *BLK%* if advanced stats are included |
| *TOV*  | Turnovers | Replaced by *TOV%* if advanced stats are included |
| *PF*   | Personal fouls |

Advanced statistics that are optionally included in the analysis:

| Stat  | Acronym | Definition |
|-------|---------|-------------|
| *PER* | Player Efficiency Rating | Per-minute rating of a player's overall performance with respect to an average player, see [here](https://www.basketball-reference.com/about/per.html)
| *ORB%* | Offensive Rebound Percentage | Estimate of the percentage of available offensive rebounds a player grabbed while he was on the floor: *ORB%* = 100 * (*ORB* * (*Tm MP* / 5)) / (*MP* * (*Tm ORB* + *Opp DRB*)) |
| *DRB%* | Defensive Rebound Percentage | Estimate of the percentage of available defensive rebounds a player grabbed while he was on the floor: *ODRB%* = 100 * (*DRB* * (*Tm MP* / 5)) / (*MP* * (*Tm DRB* + *Opp ORB*)) |
| *AST%* | Assist Percentage | Estimate of the percentage of teammate field goals a player assisted while he was on the floor: *AST%* = 100 * *AST* / (((*MP* / (*Tm MP* / 5)) * *Tm FG*) - *FG*) |
| *STL%* | Steal Percentage | Estimate of the percentage of opponent possessions that end with a steal by the player while he was on the floor: *STL%* = 100 * (*STL* * (*Tm MP* / 5)) / (*MP* * *Opp Poss*) |
| *BLK%* | Block Percentage | Estimate of the percentage of opponent 2P field goal attempts blocked by the player while he was on the floor: *BLK%* = 100 * (*BLK* * (*Tm MP* / 5)) / (*MP* * (*Opp FGA* - *Opp 3PA*)) |
| *TOV%* | Turnover Percentage | Estimate of turnovers per 100 plays: *TOV%* = 100 * *TOV* / (*FGA* + 0.44 * *FTA* + *TOV*) |
| *USG%* | Usage Percentage | Estimate of the percentage of team plays used by a player while he was on the floor: *USG%* = 100 * ((*FGA* + 0.44 * *FTA* + *TOV*) * (*Tm MP* / 5)) / (*MP* * (*Tm FGA* + 0.44 * *Tm FTA* + *Tm TOV*)) |
| *OWS* | Offensive Win Shares | See [here](https://www.basketball-reference.com/about/ws.html)
| *DWS* | Defensive Win Shares | See [here](https://www.basketball-reference.com/about/ws.html)
| *OBPM* | Offensive Box Plus/Minus | OBPM is a per-100-possession stat, see [here](https://www.basketball-reference.com/about/bpm.html)
| *DBPM* | Defensive Box Plus/Minus | DBPM is a per-100-possession stat, see [here](https://www.basketball-reference.com/about/bpm.html)
| *VORP* | Value Over Replacement Player | VORP is a per-100-possession stat, see [here](https://www.basketball-reference.com/about/bpm.html) |

## Machine Learning

The following ML classification algorithms are included in the analysis to predict the NBA All-Stars for a given year:

1. Logistic Regression Classifier
2. Nearest Neighbours Classifier
3. Linear Support Vector Machine Classifier
4. Decision Tree Classifier
5. Random Forest Classifier
6. Extra Trees Classifier
7. Gradient Tree Boosting Classifier
8. Ada Boost Classifier
9. Neural Network Classifier
10. Quadratic Discriminant Analysis Classifier
11. Gaussian Naive Bayes Classifier
12. Gaussian Process Classifier

## Analysis

To run the analysis:

- Choose the year you want to use to run some validation tests (*validation_year*) and the year you want to predict (*predicion_year*), both in range 2000-2018. The years that are not selected for validation tests and prediction are used to train the model.
- Choose whether you want to include advanced player statistics (e.g. *PER*, *VORP*, etc.) in the model or not.
- Choose a ML classifier from the list above.
