# Predicting the NBA All-Stars with Machine Learning

The goal of this analysis is to predict the NBA All-Stars for a given year, based on NBA player data and All-Star selections in other years. This is accomplished by applying several machine learning classification algorithms on player performance statistics per season. The analysis is based on the [Scikit-learn](http://scikit-learn.org) machine learning package, NBA player data are taken from [basketball-reference.com](https://www.basketball-reference.com). 

## NBA player data:

NBA player data from 2000-2018 from [basketball-reference.com](https://www.basketball-reference.com) have been saved as csv-files in the **data** directory using the scraper functions in [*NBAanalysissetup.py*](NBAanalysissetup.py). You can use [*Basketball_Reference_scraper.py*](Basketball_Reference_scraper.py) to scrape new data.

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
