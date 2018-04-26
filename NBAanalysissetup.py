#!/usr/local/bin/python3 -tt
"""
NBA analysis setup.

Change the following directory according to your system setup:
"""

myNBAanalysisdir  = "${HOME}/Programming/github_reps/NBA_analysis/" # Your NBA analysis directory

"""
Author: Gordon Lim
Last Edit: 26 Apr 2018
"""

import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re

global NBAanalysisdir    

NBAanalysisdir = os.path.expandvars(myNBAanalysisdir)
    
if not os.path.isdir(NBAanalysisdir):
    print("--- ERROR: {} does not exist - EXIT".format(NBAanalysisdir))
    exit()

class MyModel:
    """
    Container class to store ML estimators and some estimator attributes
    """
    
    def __init__(self, estimator_name, estimator):

        self.estimator_name = estimator_name
        self.estimator      = estimator
        
        self.PRE_cvgroups = []
        self.REC_cvgroups = []
        self.F1_cvgroups  = []
        self.ACC_cvgroups = []    
        self.FPR_cvgroups = []

        self.FPR_ROC_cvgroups = []
        self.TPR_ROC_cvgroups = []
        self.AUC_ROC_cvgroups = []
        
        self.PRE_PR_cvgroups  = []
        self.REC_PR_cvgroups  = []
        self.AUC_PR_cvgroups  = []

        self.cv_evs          = []
        self.cv_importances  = []
        self.cv_coefficients = []
        self.cv_y_test       = []
        self.cv_y_score      = []

        self.fpr_linspace       = 0
        self.mean_tpr_linspaces = 0
        self.rec_linspace       = 0
        self.mean_pre_linspaces = 0
        
    def reset(self):
        
        self.PRE_cvgroups[:] = []
        self.REC_cvgroups[:] = []
        self.F1_cvgroups[:]  = []
        self.ACC_cvgroups[:] = []    
        self.FPR_cvgroups[:] = []

        self.FPR_ROC_cvgroups[:] = []
        self.TPR_ROC_cvgroups[:] = []
        self.AUC_ROC_cvgroups[:] = []
        
        self.PRE_PR_cvgroups[:]  = []
        self.REC_PR_cvgroups[:]  = []
        self.AUC_PR_cvgroups[:]  = []

        self.cv_evs[:]          = []        
        self.cv_importances[:]  = []
        self.cv_coefficients[:] = []
        self.cv_y_test[:]       = []
        self.cv_y_score[:]      = []

    def set_CM(self, CM):
        self.CM = MyCM(CM)

    def set_y_truth(self, y_truth):
        self.y_truth = y_truth

    def set_y_prediction(self, y_prediction):
        self.y_prediction = y_prediction
        
    def set_y_score(self, y_score):
        self.y_score = y_score
        
    def set_y_score_calib(self, y_score_calib):
        self.y_score_calib = y_score_calib
        
        
class MyCM():
    """
    Confusion Matrix helper class
    """
    
    def __init__(self, CM):

        self.CM = CM

        self.TN = CM[0,0] # defined as: 0 = negative, 1 = positive
        self.FN = CM[1,0] # defined as: 0 = negative, 1 = positive
        self.FP = CM[0,1] # defined as: 0 = negative, 1 = positive
        self.TP = CM[1,1] # defined as: 0 = negative, 1 = positive
    
        self.TOT = self.TP + self.FP + self.FN + self.TN

        self.precision = self.TP/(self.TP+self.FP)  if (self.TP+self.FP != 0) else 0
        self.recall    = self.TP/(self.TP+self.FN)  if (self.TP+self.FN != 0) else 0
        self.fpr       = self.FP/(self.FP+self.TN)  if (self.FP+self.TN != 0) else 0
        self.accuracy  = (self.TP+self.TN)/self.TOT if (self.TOT        != 0) else 0
        self.tpr = self.recall
        self.f1 = 2*self.precision*self.recall/(self.precision+self.recall) \
            if (self.precision+self.recall != 0) else 0


def loaddata_allyears(train_years, test_year, includeadvancedstats, target):
    """
    Function that loads NBA data from csv-files for set of years
    """
    
    # Load training year data into df_train:

    dfs = []

    for train_year in train_years:
        
        print("--> Loading train year {}-{} ...".format(train_year-1, train_year))
        df = loaddata_singleyear(train_year, includeadvancedstats, target)
        dfs.append(df)

    df_train = pd.concat(dfs)
    
    # Load test data into df_test:

    print("--> Loading test  year {}-{} ...".format(test_year-1, test_year))
    df_test = loaddata_singleyear(test_year, includeadvancedstats, target)
    
    return df_train, df_test


def loaddata_singleyear(year, includeadvancedstats, target):
    """
    Function that loads NBA data from csv-files for one particular year
    """

    NBA_playerstats_csvfilename = NBAanalysisdir + 'data/NBA_totals_{}-{}.csv'.format(year-1, year)
        
    if not os.path.isfile(NBA_playerstats_csvfilename):
        print("--- ERROR: {} does not exist - EXIT".format(NBA_playerstats_csvfilename))
        exit()
    
    df = pd.read_csv(NBA_playerstats_csvfilename)

    if includeadvancedstats:
    
        NBA_playerstats_advanced_csvfilename = NBAanalysisdir + 'data/NBA_advanced_{}-{}.csv'.format(year-1, year)
    
        if not os.path.isfile(NBA_playerstats_advanced_csvfilename):
            print("*** loaddata_singleyear ERROR: {} does not exist - EXIT")
            exit()
    
        df2 = pd.read_csv(NBA_playerstats_advanced_csvfilename)
        #df2.drop(df2.columns[[19, 24]], inplace=True, axis=1)    # remove empty columns
        df2.drop(['Pos', 'Age', 'G', 'MP'], inplace=True, axis=1) # remove columns already included in regular stats csv

        df = pd.merge(df, df2, how='left', left_on=['Player', 'Tm'], right_on=['Player', 'Tm'])

    # Clean player names:

    #df = cleanplayernames(df)

    # Compress extra rows for players with more than one row: 

    df = compress_multirowplayers(df)
    
    # Add team statistics:

    df = add_team_columns(year, df)
    
    # Add target selection statistic:

    if (target == 'allstar'):
        df = add_AllStar_column(year, df)
    elif (target == 'MVP'):
        df = add_MVP_column(year, df)
    elif (target == 'ROY'):
        df = add_ROY_column(year, df)
    elif (target == 'DPOY'):
        df = add_DPOY_column(year, df)
    elif (target == 'SMOY'):
        df = add_SMOY_column(year, df)
    else:
        print("*** loaddata_singleyear ERROR: Unknown target - EXIT")
        exit()
        
    # Add YEAR for cross-validation groups:

    df['YEAR'] = year
    df['YEAR'] = df['YEAR'].astype('int64')

    #print(df.head())    
    #print(df.shape)

    return df


def add_EFF_column(df):
    """
    Function that adds the EFF statistic of players as an extra column to a dataframe
    """

    # EFF is defined as: (PTS + TRB + AST + STL + BLK − (FGA - FG) − (FTA - FT) - TOV) / G):

    df['EFF'] = (df['PTS']    + df['TRB'] + df['AST']    + df['STL'] + df['BLK'] + \
                 df['FGA']*-1 + df['FG']  + df['FTA']*-1 + df['FT']  + df['TOV']*-1) / df['G']

    return 0


def getTW(TeamWins_dict, team_acronym):
    """
    Function that returns team wins based on team acronym
    """

    fullteamname, _ = team_info(team_acronym)

    TW = TeamWins_dict[fullteamname]

    return TW

    
def getTC(team_acronym):
    """
    Function that returns conference based on team acronym
    """
    
    _, TC = team_info(team_acronym)

    return TC


def add_team_columns(year, df):
    """
    Function that adds columns with team statistics to a dataframe, based on 
    which team the players in the dataframe played for per season
    """

    NBA_teammisc_csvfilename = NBAanalysisdir + 'data/NBA_teammisc_{}-{}.csv'.format(year-1, year)

    if not os.path.isfile(NBA_teammisc_csvfilename):
        print("--- ERROR: {} does not exist - EXIT")
        exit()

    df_tm = pd.read_csv(NBA_teammisc_csvfilename)

    df_tm = df_tm.drop(['Age', 'L', 'PW', 'PL', 'MOV', 'SOS', 'SRS', 'ORtg', 'DRtg', 'Pace', 'FTr', '3PAr', 'TS%', 'eFG%', \
                        'TOV%', 'ORB%', 'FT/FGA', 'OPP_eFG%', 'OPP_TOV%', 'OPP_DRB%', 'OPP_FT/FGA', 'Arena', 'Att', 'Att/G'], axis=1)

    TeamWins_dict = {}
    for index, row in df_tm.iterrows():
        TeamWins_dict[row['Team']] = row['W']    

    df['TW'] = df.apply(lambda row: getTW(TeamWins_dict, row['Tm']), axis=1)    
        
    df['TW'] = df['TW'].astype('int64')

    df['TC'] = df.apply(lambda row: getTC(row['Tm']), axis=1)
        
    return df


def add_AllStar_column(year, df):
    """
    Function that adds the All-Star selection statistic of players as an extra column of binary values to a dataframe
    """

    NBA_allstars_csvfilename = NBAanalysisdir + 'data/NBA_allstars_{}-{}.csv'.format(year-1, year)

    if not os.path.isfile(NBA_allstars_csvfilename):
        print("--- ERROR: {} does not exist - EXIT")
        exit()

    df_as = pd.read_csv(NBA_allstars_csvfilename)

    df_as = df_as.drop(['Tm','MP','FG','FGA','3P','3PA','FT','FTA','ORB','DRB','TRB',\
                        'AST','STL','BLK','TOV','PF','PTS','FG%','3P%','FT%'], axis=1)

    df_as['AS'] = 1 # Set All-Star label
    
    #df = pd.merge(df, df_as, how='left', left_on=['Player'], right_on=['Starters'])
    df = pd.merge(df, df_as, how='left', left_on=['Player'], right_on=['Player'])
    #df = df.drop(['Starters'], axis=1) 
    
    values = {'AS': 0}
    df.fillna(value=values, inplace=True) # Set non-All-Star label

    df['AS'] = df['AS'].astype('int64')
    
    return df


def add_MVP_column(year, df):
    """
    Function that adds the MVP voting statistic of players as an extra column of floats to a dataframe
    """

    NBA_MVP_csvfilename = NBAanalysisdir + 'data/NBA_MVP_{}-{}.csv'.format(year-1, year)

    if not os.path.isfile(NBA_MVP_csvfilename):
        print("--- ERROR: {} does not exist - EXIT")
        exit()

    df_as = pd.read_csv(NBA_MVP_csvfilename)

    df_as = df_as.drop(['Age', 'Tm', 'First', 'PtsWon', 'PtsMax', 'G', 'MP', 'PTS', 'TRB',
                        'AST', 'STL', 'BLK', 'FG%', '3P%', 'FT%', 'WS', 'WS/48'], axis=1)

    df = pd.merge(df, df_as, how='left', left_on=['Player'], right_on=['Player'])
    
    df = df.rename(index=str, columns={"Share": "AVS"})

    values = {'AVS': 0}
    df.fillna(value=values, inplace=True) # Set MVP Vote Share to 0 for players without MVP votes
    
    return df


def add_ROY_column(year, df):
    """
    Function that adds the ROY voting statistic of players as an extra column of floats to a dataframe
    """

    NBA_rookies_csvfilename = NBAanalysisdir + 'data/NBA_rookies_{}-{}.csv'.format(year-1, year)

    if not os.path.isfile(NBA_rookies_csvfilename):
        print("--- ERROR: {} does not exist - EXIT")
        exit()

    df_as = pd.read_csv(NBA_rookies_csvfilename)

    df_as = df_as.drop(['Age', 'Yrs', 'G', 'MP', 'FG', 'FGA', '3P', '3PA', 'FT', 'FTA', 'ORB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'FG%', '3P%', 'FT%', 'MP/G', 'PTS/G', 'TRB/G', 'AST/G'], axis=1)

    df = pd.merge(df, df_as, how='right', left_on=['Player'], right_on=['Player'])

    NBA_ROY_csvfilename = NBAanalysisdir + 'data/NBA_ROY_{}-{}.csv'.format(year-1, year)

    if not os.path.isfile(NBA_ROY_csvfilename):
        print("--- ERROR: {} does not exist - EXIT")
        exit()

    df_as = pd.read_csv(NBA_ROY_csvfilename)

    df_as = df_as.drop(['Age', 'Tm', 'First', 'PtsWon', 'PtsMax', 'G', 'MP', 'PTS', 'TRB',
                        'AST', 'STL', 'BLK', 'FG%', '3P%', 'FT%', 'WS', 'WS/48'], axis=1)

    df = pd.merge(df, df_as, how='left', left_on=['Player'], right_on=['Player'])
    
    df = df.rename(index=str, columns={"Share": "AVS"})

    values = {'AVS': 0}
    df.fillna(value=values, inplace=True) # Set ROY Vote Share to 0 for players without ROY votes
    
    return df


def add_DPOY_column(year, df):
    """
    Function that adds the DPOY voting statistic of players as an extra column of floats to a dataframe
    """

    NBA_DPOY_csvfilename = NBAanalysisdir + 'data/NBA_DPOY_{}-{}.csv'.format(year-1, year)

    if not os.path.isfile(NBA_DPOY_csvfilename):
        print("--- ERROR: {} does not exist - EXIT")
        exit()

    df_as = pd.read_csv(NBA_DPOY_csvfilename)

    df_as = df_as.drop(['Age', 'Tm', 'First', 'PtsWon', 'PtsMax', 'G', 'MP', 'PTS', 'TRB',
                        'AST', 'STL', 'BLK', 'FG%', '3P%', 'FT%', 'WS', 'WS/48'], axis=1)

    df = pd.merge(df, df_as, how='left', left_on=['Player'], right_on=['Player'])
    
    df = df.rename(index=str, columns={"Share": "AVS"})

    values = {'AVS': 0}
    df.fillna(value=values, inplace=True) # Set DPOY Vote Share to 0 for players without DPOY votes
    
    return df


def add_SMOY_column(year, df):
    """
    Function that adds the SMOY voting statistic of players as an extra column of floats to a dataframe
    """

    NBA_SMOY_csvfilename = NBAanalysisdir + 'data/NBA_SMOY_{}-{}.csv'.format(year-1, year)

    if not os.path.isfile(NBA_SMOY_csvfilename):
        print("--- ERROR: {} does not exist - EXIT")
        exit()

    df_as = pd.read_csv(NBA_SMOY_csvfilename)

    df_as = df_as.drop(['Age', 'Tm', 'First', 'PtsWon', 'PtsMax', 'G', 'MP', 'PTS', 'TRB',
                        'AST', 'STL', 'BLK', 'FG%', '3P%', 'FT%', 'WS', 'WS/48'], axis=1)

    df = pd.merge(df, df_as, how='left', left_on=['Player'], right_on=['Player'])
    
    df = df.rename(index=str, columns={"Share": "AVS"})

    values = {'AVS': 0}
    df.fillna(value=values, inplace=True) # Set SMOY Vote Share to 0 for players without SMOY votes
    
    return df


def cleanplayernames(df):
    """
    #This function cleans up player names by removing the part that starts with "\":
    This function cleans up player names by removing a trailing asterix from a name if present:
    """

    #df['Player'].replace(to_replace=r'(\\[\w\d]+$)', value='', regex=True, inplace=True)
    df['Player'].replace(to_replace=r'(\*$)', value='', regex=True, inplace=True)

    return df


def compress_multirowplayers(df):
    """
    This function removes extra rows for players with more than one row (because they switched teams during the season): For players with more than one row, keep only row with 'Tm' == 'TOT' and replace 'Tm' value with most recent team id:
    """

    indices_of_rows_toberemoved = []
    player_team_dict = {}

    previous_player_id = ''
    
    for index, row in df.iterrows():
        player_id = row['Player']
        if (player_id == previous_player_id):
            if (row['Tm'] != 'TOT'):
                indices_of_rows_toberemoved.append(index)
                player_team_dict[player_id] = row['Tm'] # this works because the last team in the list is the player's most recent team 
        previous_player_id = player_id

    for index in indices_of_rows_toberemoved:
        df.drop(index, inplace=True)

    for index, row in df.iterrows():
        for key, value in player_team_dict.items():
            if (row['Player'] == key):
                df.at[index, 'Tm'] = value

    return df


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the (normalized) Confusion Matrix.
    See http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
            
    plt.ylabel('Truth')
    plt.xlabel('Prediction')
    plt.grid(False)
    plt.tight_layout()

    
def NBA_totals_scraper(year):
    """
    NBA totals data scraper function
    """

    import requests
    import csv
    from bs4 import BeautifulSoup

    out_path = NBAanalysisdir + 'data/NBA_totals_{}-{}.csv'.format(year-1, year)
    csv_file = open(out_path, 'w')
    csv_writer = csv.writer(csv_file)

    features = ['Player', 'Pos', 'Age', 'Tm', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', \
                '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']
    
    csv_writer.writerow(features)
    
    URL = 'https://www.basketball-reference.com/leagues/NBA_{}_totals.html'.format(year)

    print("--- Scraping totals data {}-{}...".format(year-1, year))
    
    r = requests.get(URL)
    soup = BeautifulSoup(r.text, "html5lib")

    table = soup.find(id="all_totals_stats")
    cells = table.find_all('td')

    ncolumns = len(features)
    
    Player  = [cells[i].getText() for i in range( 0, len(cells), ncolumns)]
    Pos     = [cells[i].getText() for i in range( 1, len(cells), ncolumns)]
    Age     = [cells[i].getText() for i in range( 2, len(cells), ncolumns)]
    Tm      = [cells[i].getText() for i in range( 3, len(cells), ncolumns)]
    G       = [cells[i].getText() for i in range( 4, len(cells), ncolumns)]
    GS      = [cells[i].getText() for i in range( 5, len(cells), ncolumns)]
    MP      = [cells[i].getText() for i in range( 6, len(cells), ncolumns)]
    FG      = [cells[i].getText() for i in range( 7, len(cells), ncolumns)]
    FGA     = [cells[i].getText() for i in range( 8, len(cells), ncolumns)]
    FGP     = [cells[i].getText() for i in range( 9, len(cells), ncolumns)]
    THP     = [cells[i].getText() for i in range(10, len(cells), ncolumns)]
    THPA    = [cells[i].getText() for i in range(11, len(cells), ncolumns)]
    THPP    = [cells[i].getText() for i in range(12, len(cells), ncolumns)]
    TWP     = [cells[i].getText() for i in range(13, len(cells), ncolumns)]
    TWPA    = [cells[i].getText() for i in range(14, len(cells), ncolumns)]
    TWPP    = [cells[i].getText() for i in range(15, len(cells), ncolumns)]
    EFGP    = [cells[i].getText() for i in range(16, len(cells), ncolumns)]
    FT      = [cells[i].getText() for i in range(17, len(cells), ncolumns)]
    FTA     = [cells[i].getText() for i in range(18, len(cells), ncolumns)]
    FTP     = [cells[i].getText() for i in range(19, len(cells), ncolumns)]
    ORB     = [cells[i].getText() for i in range(20, len(cells), ncolumns)]
    DRB     = [cells[i].getText() for i in range(21, len(cells), ncolumns)]
    TRB     = [cells[i].getText() for i in range(22, len(cells), ncolumns)]
    AST     = [cells[i].getText() for i in range(23, len(cells), ncolumns)]
    STL     = [cells[i].getText() for i in range(24, len(cells), ncolumns)]
    BLK     = [cells[i].getText() for i in range(25, len(cells), ncolumns)]
    TOV     = [cells[i].getText() for i in range(26, len(cells), ncolumns)]
    PF      = [cells[i].getText() for i in range(27, len(cells), ncolumns)]
    PTS     = [cells[i].getText() for i in range(28, len(cells), ncolumns)]

    Player = [i.replace('*', '') for i in Player] # Remove possible asterix from player name
    
    for i in range(0, int(len(cells) / ncolumns)):
        row = [Player[i], Pos[i], Age[i], Tm[i], G[i], GS[i], MP[i], FG[i], FGA[i], FGP[i], THP[i], THPA[i], THPP[i], TWP[i], TWPA[i], \
               TWPP[i], EFGP[i], FT[i], FTA[i], FTP[i], ORB[i], DRB[i], TRB[i], AST[i], STL[i], BLK[i], TOV[i], PF[i], PTS[i]]
        csv_writer.writerow(row)

        
def NBA_advanced_scraper(year):
    """
    NBA advanced data scraper function
    """

    import requests
    import csv
    from bs4 import BeautifulSoup

    out_path = NBAanalysisdir + 'data/NBA_advanced_{}-{}.csv'.format(year-1, year)
    csv_file = open(out_path, 'w')
    csv_writer = csv.writer(csv_file)

    features = ['Player', 'Pos', 'Age', 'Tm', 'G', 'MP', 'PER', 'TS%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%', \
                'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'OWS', 'DWS', 'WS', 'WS/48', 'OBPM', 'DBPM', 'BPM', 'VORP']
        
    csv_writer.writerow(features)
    
    URL = 'https://www.basketball-reference.com/leagues/NBA_{}_advanced.html'.format(year)

    print("--- Scraping advanced data {}-{}...".format(year-1, year))
    
    r = requests.get(URL)
    soup = BeautifulSoup(r.text, "html5lib")

    table = soup.find(id="all_advanced_stats")
    cells = table.find_all('td')

    ncolumns = len(features) + 2 # plus 2 because there are two columns missing!

    Player  = [cells[i].getText() for i in range( 0, len(cells), ncolumns)]
    Pos     = [cells[i].getText() for i in range( 1, len(cells), ncolumns)]
    Age     = [cells[i].getText() for i in range( 2, len(cells), ncolumns)]
    Tm      = [cells[i].getText() for i in range( 3, len(cells), ncolumns)]
    G       = [cells[i].getText() for i in range( 4, len(cells), ncolumns)]
    MP      = [cells[i].getText() for i in range( 5, len(cells), ncolumns)]
    PER     = [cells[i].getText() for i in range( 6, len(cells), ncolumns)]
    TSP     = [cells[i].getText() for i in range( 7, len(cells), ncolumns)]
    TPAr    = [cells[i].getText() for i in range( 8, len(cells), ncolumns)]
    FTr     = [cells[i].getText() for i in range( 9, len(cells), ncolumns)]
    ORBP    = [cells[i].getText() for i in range(10, len(cells), ncolumns)]
    DRBP    = [cells[i].getText() for i in range(11, len(cells), ncolumns)]
    TRBP    = [cells[i].getText() for i in range(12, len(cells), ncolumns)]
    ASTP    = [cells[i].getText() for i in range(13, len(cells), ncolumns)]
    STLP    = [cells[i].getText() for i in range(14, len(cells), ncolumns)]
    BLKP    = [cells[i].getText() for i in range(15, len(cells), ncolumns)]
    TOVP    = [cells[i].getText() for i in range(16, len(cells), ncolumns)]
    USGP    = [cells[i].getText() for i in range(17, len(cells), ncolumns)]
    OWS     = [cells[i].getText() for i in range(19, len(cells), ncolumns)] # 18 is empty!
    DWS     = [cells[i].getText() for i in range(20, len(cells), ncolumns)]
    WS      = [cells[i].getText() for i in range(21, len(cells), ncolumns)]
    WS48    = [cells[i].getText() for i in range(22, len(cells), ncolumns)]
    OBPM    = [cells[i].getText() for i in range(24, len(cells), ncolumns)] # 23 is empty!
    DBPM    = [cells[i].getText() for i in range(25, len(cells), ncolumns)]
    BPM     = [cells[i].getText() for i in range(26, len(cells), ncolumns)]
    VORP    = [cells[i].getText() for i in range(27, len(cells), ncolumns)]

    Player = [i.replace('*', '') for i in Player] # Remove possible asterix from player name
    
    for i in range(0, int(len(cells) / ncolumns)):
        row = [Player[i], Pos[i], Age[i], Tm[i], G[i], MP[i], PER[i], TSP[i], TPAr[i], FTr[i], ORBP[i], DRBP[i], TRBP[i], \
               ASTP[i], STLP[i], BLKP[i], TOVP[i], USGP[i], OWS[i], DWS[i], WS[i], WS48[i], OBPM[i], DBPM[i], BPM[i], VORP[i]]
        csv_writer.writerow(row)

    return 0

        
def NBA_AllStar_scraper(year):
    """
    NBA All-Stars data scraper function
    """

    import requests
    import csv
    from bs4 import BeautifulSoup
    
    out_path = NBAanalysisdir + 'data/NBA_allstars_{}-{}.csv'.format(year-1, year)
    csv_file = open(out_path, 'w')
    csv_writer = csv.writer(csv_file)

    features = ['Player', 'Tm', 'MP', 'FG', 'FGA', '3P', '3PA', 'FT', 'FTA', 'ORB', \
                'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'FG%', '3P%', 'FT%']
    csv_writer.writerow(features)
    
    URL = 'https://www.basketball-reference.com/allstar/NBA_{}.html'.format(year)

    print("--- Scraping All-Stars data {}-{}...".format(year-1, year))
    
    r = requests.get(URL)
    soup = BeautifulSoup(r.text, "html5lib")

    if (year < 2018):
        NBA_AllStar_scraper_team(soup, "all_West", csv_writer)
        NBA_AllStar_scraper_team(soup, "all_East", csv_writer)
    elif (year == 2018):
        NBA_AllStar_scraper_team(soup, "all_Stephen", csv_writer)
        NBA_AllStar_scraper_team(soup, "all_LeBron",  csv_writer)
    else:
        print("WRONG YEAR")
        exit()
        
    return 0


def NBA_AllStar_scraper_team(soup, team, csv_writer):
    """
    NBA All-Stars data scraper for one team (i.e. EAST and WEST)
    """

    table = soup.find(id=team)
    
    players = table.find_all('th', class_="left ")
    
    cells = table.find_all('td')

    ncolumns = 20

    Tm  = [cells[i].getText() for i in range( 0, len(cells), ncolumns)]
    MP  = [cells[i].getText() for i in range( 1, len(cells), ncolumns)]
    FG  = [cells[i].getText() for i in range( 2, len(cells), ncolumns)]
    FGA = [cells[i].getText() for i in range( 3, len(cells), ncolumns)]
    TP  = [cells[i].getText() for i in range( 4, len(cells), ncolumns)]
    TPA = [cells[i].getText() for i in range( 5, len(cells), ncolumns)]
    FT  = [cells[i].getText() for i in range( 6, len(cells), ncolumns)]
    FTA = [cells[i].getText() for i in range( 7, len(cells), ncolumns)]
    ORB = [cells[i].getText() for i in range( 8, len(cells), ncolumns)]
    DRB = [cells[i].getText() for i in range( 9, len(cells), ncolumns)]
    TRB = [cells[i].getText() for i in range(10, len(cells), ncolumns)]
    AST = [cells[i].getText() for i in range(11, len(cells), ncolumns)]
    STL = [cells[i].getText() for i in range(12, len(cells), ncolumns)]
    BLK = [cells[i].getText() for i in range(13, len(cells), ncolumns)]
    TOV = [cells[i].getText() for i in range(14, len(cells), ncolumns)]
    PF  = [cells[i].getText() for i in range(15, len(cells), ncolumns)]
    PTS = [cells[i].getText() for i in range(16, len(cells), ncolumns)]
    FGP = [cells[i].getText() for i in range(17, len(cells), ncolumns)]
    TPP = [cells[i].getText() for i in range(18, len(cells), ncolumns)]
    FTP = [cells[i].getText() for i in range(19, len(cells), ncolumns)]
    
    for i in range(0, int(len(cells) / ncolumns) - 1): # skip last line
        row = [players[i].getText(), Tm[i], MP[i], FG[i], FGA[i], TP[i], TPA[i], FT[i], FTA[i], ORB[i], \
               DRB[i], TRB[i], AST[i], STL[i], BLK[i], TOV[i], PF[i], PTS[i], FGP[i], TPP[i], FTP[i]]
        csv_writer.writerow(row)

    return 0


def NBA_teamstats_scraper(year):
    """
    NBA team stats data scraper function
    """

    import requests
    import csv
    from bs4 import BeautifulSoup

    out_path = NBAanalysisdir + 'data/NBA_teamstats_{}-{}.csv'.format(year-1, year)
    csv_file = open(out_path, 'w')
    csv_writer = csv.writer(csv_file)

    features = ['Team', 'G', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', \
                'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']

    csv_writer.writerow(features)
    
    URL = 'https://www.basketball-reference.com/leagues/NBA_{}.html'.format(year)

    print("--- Scraping team stats data {}-{}...".format(year-1, year))
    
    r = requests.get(URL)
    soup = BeautifulSoup(r.text, "html5lib")

    table_text = soup.find(text=re.compile('table class="sortable stats_table" id="team-stats-base"'))
    table_soup = BeautifulSoup(table_text, "html5lib")
    cells = table_soup.find_all('td')

    ncolumns = len(features)
    
    Team = [cells[i].getText() for i in range( 0, len(cells), ncolumns)]
    G    = [cells[i].getText() for i in range( 1, len(cells), ncolumns)]
    MP   = [cells[i].getText() for i in range( 2, len(cells), ncolumns)]
    FG   = [cells[i].getText() for i in range( 3, len(cells), ncolumns)]
    FGA  = [cells[i].getText() for i in range( 4, len(cells), ncolumns)]
    FGP  = [cells[i].getText() for i in range( 5, len(cells), ncolumns)]
    THP  = [cells[i].getText() for i in range( 6, len(cells), ncolumns)]
    THPA = [cells[i].getText() for i in range( 7, len(cells), ncolumns)]
    THPP = [cells[i].getText() for i in range( 8, len(cells), ncolumns)]
    TWP  = [cells[i].getText() for i in range( 9, len(cells), ncolumns)]
    TWPA = [cells[i].getText() for i in range(10, len(cells), ncolumns)]
    TWPP = [cells[i].getText() for i in range(11, len(cells), ncolumns)]
    FT   = [cells[i].getText() for i in range(12, len(cells), ncolumns)]
    FTA  = [cells[i].getText() for i in range(13, len(cells), ncolumns)]
    FTP  = [cells[i].getText() for i in range(14, len(cells), ncolumns)]
    ORB  = [cells[i].getText() for i in range(15, len(cells), ncolumns)]
    DRB  = [cells[i].getText() for i in range(16, len(cells), ncolumns)]
    TRB  = [cells[i].getText() for i in range(17, len(cells), ncolumns)]
    AST  = [cells[i].getText() for i in range(18, len(cells), ncolumns)]
    STL  = [cells[i].getText() for i in range(19, len(cells), ncolumns)]
    BLK  = [cells[i].getText() for i in range(20, len(cells), ncolumns)]
    TOV  = [cells[i].getText() for i in range(21, len(cells), ncolumns)]
    PF   = [cells[i].getText() for i in range(22, len(cells), ncolumns)]
    PTS  = [cells[i].getText() for i in range(23, len(cells), ncolumns)]

    Team = [i.replace('*', '') for i in Team] # Remove possible asterix from team name
    
    for i in range(0, int(len(cells) / ncolumns) - 1): # Skip last line
        row = [Team[i], G[i], MP[i], FG[i], FGA[i], FGP[i], THP[i], THPA[i], THPP[i], TWP[i], TWPA[i], TWPP[i], \
               FT[i], FTA[i], FTP[i], ORB[i], DRB[i], TRB[i], AST[i], STL[i], BLK[i], TOV[i], PF[i], PTS[i]]
        csv_writer.writerow(row)

    return 0


def NBA_teammisc_scraper(year):
    """
    NBA team miscellaneous stats data scraper function
    """

    import requests
    import csv
    from bs4 import BeautifulSoup

    out_path = NBAanalysisdir + 'data/NBA_teammisc_{}-{}.csv'.format(year-1, year)
    csv_file = open(out_path, 'w')
    csv_writer = csv.writer(csv_file)

    features = ['Team', 'Age', 'W', 'L', 'PW', 'PL', 'MOV', 'SOS', 'SRS', 'ORtg', 'DRtg', 'Pace', 'FTr', '3PAr', 'TS%', \
                'eFG%', 'TOV%', 'ORB%', 'FT/FGA', 'OPP_eFG%', 'OPP_TOV%', 'OPP_DRB%', 'OPP_FT/FGA', 'Arena', 'Att', 'Att/G']
        
    csv_writer.writerow(features)
    
    URL = 'https://www.basketball-reference.com/leagues/NBA_{}.html'.format(year)

    print("--- Scraping team miscellaneous stats data {}-{}...".format(year-1, year))
    
    r = requests.get(URL)
    soup = BeautifulSoup(r.text, "html5lib")

    table_text = soup.find(text=re.compile('table class="sortable stats_table" id="misc_stats"'))
    table_soup = BeautifulSoup(table_text, "html5lib")
    cells = table_soup.find_all('td')

    ncolumns = len(features)
    
    Team     = [cells[i].getText() for i in range( 0, len(cells), ncolumns)]
    Age      = [cells[i].getText() for i in range( 1, len(cells), ncolumns)]
    W        = [cells[i].getText() for i in range( 2, len(cells), ncolumns)]
    L        = [cells[i].getText() for i in range( 3, len(cells), ncolumns)]
    PW       = [cells[i].getText() for i in range( 4, len(cells), ncolumns)]
    PL       = [cells[i].getText() for i in range( 5, len(cells), ncolumns)]
    MOV      = [cells[i].getText() for i in range( 6, len(cells), ncolumns)]
    SOS      = [cells[i].getText() for i in range( 7, len(cells), ncolumns)]
    SRS      = [cells[i].getText() for i in range( 8, len(cells), ncolumns)]
    ORtg     = [cells[i].getText() for i in range( 9, len(cells), ncolumns)]
    DRtg     = [cells[i].getText() for i in range(10, len(cells), ncolumns)]
    Pace     = [cells[i].getText() for i in range(11, len(cells), ncolumns)]
    FTr      = [cells[i].getText() for i in range(12, len(cells), ncolumns)]
    TPAr     = [cells[i].getText() for i in range(13, len(cells), ncolumns)]
    TSP      = [cells[i].getText() for i in range(14, len(cells), ncolumns)]
    EFGP     = [cells[i].getText() for i in range(15, len(cells), ncolumns)]
    TOVP     = [cells[i].getText() for i in range(16, len(cells), ncolumns)]
    ORBP     = [cells[i].getText() for i in range(17, len(cells), ncolumns)]
    FTPFGA   = [cells[i].getText() for i in range(18, len(cells), ncolumns)]
    O_EFGP   = [cells[i].getText() for i in range(19, len(cells), ncolumns)]
    O_TOVP   = [cells[i].getText() for i in range(20, len(cells), ncolumns)]
    O_DRBP   = [cells[i].getText() for i in range(21, len(cells), ncolumns)]
    O_FTPFGA = [cells[i].getText() for i in range(22, len(cells), ncolumns)]
    Arena    = [cells[i].getText() for i in range(23, len(cells), ncolumns)]
    Att      = [cells[i].getText() for i in range(24, len(cells), ncolumns)]
    AttG     = [cells[i].getText() for i in range(25, len(cells), ncolumns)]

    Team = [i.replace('*', '') for i in Team] # Remove possible asterix from team name
    Att  = [i.replace(',', '') for i in Att]  # Remove comma from Attendence
    AttG = [i.replace(',', '') for i in AttG] # Remove comma from Attendence per Game
    Att  = [i.replace('"', '') for i in Att]  # Remove quotes from Attendence
    AttG = [i.replace('"', '') for i in AttG] # Remove quotes from Attendence per Game

    for i in range(0, int(len(cells) / ncolumns) - 1): # Skip last line
        row = [Team[i], Age[i], W[i], L[i], PW[i], PL[i], MOV[i], SOS[i], SRS[i], ORtg[i], DRtg[i], Pace[i], FTr[i], TPAr[i], TSP[i], \
               EFGP[i], TOVP[i], ORBP[i], FTPFGA[i], O_EFGP[i], O_TOVP[i], O_DRBP[i], O_FTPFGA[i], Arena[i], Att[i], AttG[i]]
        csv_writer.writerow(row)

    return 0


def NBA_rookies_scraper(year):
    """
    NBA rookies data scraper function
    """

    import requests
    import csv
    from bs4 import BeautifulSoup

    out_path = NBAanalysisdir + 'data/NBA_rookies_{}-{}.csv'.format(year-1, year)
    csv_file = open(out_path, 'w')
    csv_writer = csv.writer(csv_file)
    
    features = ['Player', 'Age', 'Yrs', 'G', 'MP', 'FG', 'FGA', '3P', '3PA', 'FT', 'FTA', 'ORB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'FG%', '3P%', 'FT%', 'MP/G', 'PTS/G', 'TRB/G', 'AST/G']
    
    csv_writer.writerow(features)
    
    URL = 'https://www.basketball-reference.com/leagues/NBA_{}_rookies.html'.format(year)

    print("--- Scraping rookies data {}-{}...".format(year-1, year))
    
    r = requests.get(URL)
    soup = BeautifulSoup(r.text, "html5lib")

    table = soup.find(id="rookies")
    cells = table.find_all('td')

    ncolumns = len(features)
    
    Player = [cells[i].getText() for i in range( 0, len(cells), ncolumns)]
    Age    = [cells[i].getText() for i in range( 1, len(cells), ncolumns)]
    Yrs    = [cells[i].getText() for i in range( 2, len(cells), ncolumns)]
    G      = [cells[i].getText() for i in range( 3, len(cells), ncolumns)]
    MP     = [cells[i].getText() for i in range( 4, len(cells), ncolumns)]
    FG     = [cells[i].getText() for i in range( 5, len(cells), ncolumns)]
    FGA    = [cells[i].getText() for i in range( 6, len(cells), ncolumns)]
    THP    = [cells[i].getText() for i in range( 7, len(cells), ncolumns)]
    THPA   = [cells[i].getText() for i in range( 8, len(cells), ncolumns)]
    FT     = [cells[i].getText() for i in range( 9, len(cells), ncolumns)]
    FTA    = [cells[i].getText() for i in range(10, len(cells), ncolumns)]
    ORB    = [cells[i].getText() for i in range(11, len(cells), ncolumns)]
    TRB    = [cells[i].getText() for i in range(12, len(cells), ncolumns)]
    AST    = [cells[i].getText() for i in range(13, len(cells), ncolumns)]
    STL    = [cells[i].getText() for i in range(14, len(cells), ncolumns)]
    BLK    = [cells[i].getText() for i in range(15, len(cells), ncolumns)]
    TOV    = [cells[i].getText() for i in range(16, len(cells), ncolumns)]
    PF     = [cells[i].getText() for i in range(17, len(cells), ncolumns)]
    PTS    = [cells[i].getText() for i in range(18, len(cells), ncolumns)]
    FGP    = [cells[i].getText() for i in range(19, len(cells), ncolumns)]
    THPP   = [cells[i].getText() for i in range(20, len(cells), ncolumns)]
    FTP    = [cells[i].getText() for i in range(21, len(cells), ncolumns)]
    MPG    = [cells[i].getText() for i in range(22, len(cells), ncolumns)]
    PTSG   = [cells[i].getText() for i in range(23, len(cells), ncolumns)]
    TRBG   = [cells[i].getText() for i in range(24, len(cells), ncolumns)]
    ASTG   = [cells[i].getText() for i in range(25, len(cells), ncolumns)]
    
    Player = [i.replace('*', '') for i in Player] # Remove possible asterix from player name
    
    for i in range(0, int(len(cells) / ncolumns)):
        row = [Player[i], Age[i], Yrs[i], G[i], MP[i], FG[i], FGA[i], THP[i], THPA[i], FT[i], FTA[i], ORB[i], TRB[i], AST[i], STL[i], BLK[i], TOV[i], PF[i], PTS[i], FGP[i], THPP[i], FTP[i], MPG[i], PTSG[i], TRBG[i], ASTG[i]]
        csv_writer.writerow(row)


def NBA_MVP_scraper(year):
    """
    NBA MVP voting data scraper function
    """

    import requests
    import csv
    from bs4 import BeautifulSoup

    out_path = NBAanalysisdir + 'data/NBA_MVP_{}-{}.csv'.format(year-1, year)
    csv_file = open(out_path, 'w')
    csv_writer = csv.writer(csv_file)
    
    features = ['Player', 'Age', 'Tm', 'First', 'PtsWon', 'PtsMax', 'Share',
                'G', 'MP', 'PTS', 'TRB', 'AST', 'STL', 'BLK', 'FG%', '3P%', 'FT%', 'WS', 'WS/48']
    
    csv_writer.writerow(features)
    
    URL = 'https://www.basketball-reference.com/awards/awards_{}.html'.format(year)

    print("--- Scraping MVP data {}-{}...".format(year-1, year))
    
    r = requests.get(URL)
    soup = BeautifulSoup(r.text, "html5lib")

    table = soup.find(id="div_mvp")
    cells = table.find_all('td')

    ncolumns = len(features)
    
    Player  = [cells[i].getText() for i in range( 0, len(cells), ncolumns)]
    Age     = [cells[i].getText() for i in range( 1, len(cells), ncolumns)]
    Tm      = [cells[i].getText() for i in range( 2, len(cells), ncolumns)]
    First   = [cells[i].getText() for i in range( 3, len(cells), ncolumns)]
    PtsWon  = [cells[i].getText() for i in range( 4, len(cells), ncolumns)]
    PtsMax  = [cells[i].getText() for i in range( 5, len(cells), ncolumns)]
    Share   = [cells[i].getText() for i in range( 6, len(cells), ncolumns)]
    G       = [cells[i].getText() for i in range( 7, len(cells), ncolumns)]
    MP      = [cells[i].getText() for i in range( 8, len(cells), ncolumns)]
    PTS     = [cells[i].getText() for i in range( 9, len(cells), ncolumns)]
    TRB     = [cells[i].getText() for i in range(10, len(cells), ncolumns)]
    AST     = [cells[i].getText() for i in range(11, len(cells), ncolumns)]
    STL     = [cells[i].getText() for i in range(12, len(cells), ncolumns)]
    BLK     = [cells[i].getText() for i in range(13, len(cells), ncolumns)]
    FGP     = [cells[i].getText() for i in range(14, len(cells), ncolumns)]
    TPP     = [cells[i].getText() for i in range(15, len(cells), ncolumns)]
    FTP     = [cells[i].getText() for i in range(16, len(cells), ncolumns)]
    WS      = [cells[i].getText() for i in range(17, len(cells), ncolumns)]
    WS48    = [cells[i].getText() for i in range(18, len(cells), ncolumns)]

    Player = [i.replace('*', '') for i in Player] # Remove possible asterix from player name
    
    for i in range(0, int(len(cells) / ncolumns)):
        row = [Player[i], Age[i], Tm[i], First[i], PtsWon[i], PtsMax[i], Share[i],
               G[i], MP[i], PTS[i], TRB[i], AST[i], STL[i], BLK[i], FGP[i], TPP[i], FTP[i], WS[i], WS48[i]]
        csv_writer.writerow(row)

        
def NBA_ROY_scraper(year):
    """
    NBA ROY voting data scraper function
    """

    import requests
    import csv
    from bs4 import BeautifulSoup

    out_path = NBAanalysisdir + 'data/NBA_ROY_{}-{}.csv'.format(year-1, year)
    csv_file = open(out_path, 'w')
    csv_writer = csv.writer(csv_file)

    features = ['Player', 'Age', 'Tm', 'First', 'PtsWon', 'PtsMax', 'Share',
                'G', 'MP', 'PTS', 'TRB', 'AST', 'STL', 'BLK', 'FG%', '3P%', 'FT%', 'WS', 'WS/48']
    
    csv_writer.writerow(features)
    
    URL = 'https://www.basketball-reference.com/awards/awards_{}.html'.format(year)

    print("--- Scraping ROY data {}-{}...".format(year-1, year))
    
    r = requests.get(URL)
    soup = BeautifulSoup(r.text, "html5lib")

    table = soup.find(id="div_roy")
    cells = table.find_all('td')

    ncolumns = len(features)
    
    Player  = [cells[i].getText() for i in range( 0, len(cells), ncolumns)]
    Age     = [cells[i].getText() for i in range( 1, len(cells), ncolumns)]
    Tm      = [cells[i].getText() for i in range( 2, len(cells), ncolumns)]
    First   = [cells[i].getText() for i in range( 3, len(cells), ncolumns)]
    PtsWon  = [cells[i].getText() for i in range( 4, len(cells), ncolumns)]
    PtsMax  = [cells[i].getText() for i in range( 5, len(cells), ncolumns)]
    Share   = [cells[i].getText() for i in range( 6, len(cells), ncolumns)]
    G       = [cells[i].getText() for i in range( 7, len(cells), ncolumns)]
    MP      = [cells[i].getText() for i in range( 8, len(cells), ncolumns)]
    PTS     = [cells[i].getText() for i in range( 9, len(cells), ncolumns)]
    TRB     = [cells[i].getText() for i in range(10, len(cells), ncolumns)]
    AST     = [cells[i].getText() for i in range(11, len(cells), ncolumns)]
    STL     = [cells[i].getText() for i in range(12, len(cells), ncolumns)]
    BLK     = [cells[i].getText() for i in range(13, len(cells), ncolumns)]
    FGP     = [cells[i].getText() for i in range(14, len(cells), ncolumns)]
    TPP     = [cells[i].getText() for i in range(15, len(cells), ncolumns)]
    FTP     = [cells[i].getText() for i in range(16, len(cells), ncolumns)]
    WS      = [cells[i].getText() for i in range(17, len(cells), ncolumns)]
    WS48    = [cells[i].getText() for i in range(18, len(cells), ncolumns)]

    Player = [i.replace('*', '') for i in Player] # Remove possible asterix from player name
    
    for i in range(0, int(len(cells) / ncolumns)):
        row = [Player[i], Age[i], Tm[i], First[i], PtsWon[i], PtsMax[i], Share[i],
               G[i], MP[i], PTS[i], TRB[i], AST[i], STL[i], BLK[i], FGP[i], TPP[i], FTP[i], WS[i], WS48[i]]
        csv_writer.writerow(row)


def NBA_DPOY_scraper(year):
    """
    NBA DPOY voting data scraper function
    """

    import requests
    import csv
    from bs4 import BeautifulSoup

    out_path = NBAanalysisdir + 'data/NBA_DPOY_{}-{}.csv'.format(year-1, year)
    csv_file = open(out_path, 'w')
    csv_writer = csv.writer(csv_file)

    features = ['Player', 'Age', 'Tm', 'First', 'PtsWon', 'PtsMax', 'Share',
                'G', 'MP', 'PTS', 'TRB', 'AST', 'STL', 'BLK', 'FG%', '3P%', 'FT%', 'WS', 'WS/48']
    
    csv_writer.writerow(features)
    
    URL = 'https://www.basketball-reference.com/awards/awards_{}.html'.format(year)

    print("--- Scraping DPOY data {}-{}...".format(year-1, year))
    
    r = requests.get(URL)
    soup = BeautifulSoup(r.text, "html5lib")

    table_text = soup.find(text=re.compile('table class="sortable stats_table" id="dpoy"'))
    table_soup = BeautifulSoup(table_text, "html5lib")
    cells = table_soup.find_all('td')

    ncolumns = len(features)
    
    Player  = [cells[i].getText() for i in range( 0, len(cells), ncolumns)]
    Age     = [cells[i].getText() for i in range( 1, len(cells), ncolumns)]
    Tm      = [cells[i].getText() for i in range( 2, len(cells), ncolumns)]
    First   = [cells[i].getText() for i in range( 3, len(cells), ncolumns)]
    PtsWon  = [cells[i].getText() for i in range( 4, len(cells), ncolumns)]
    PtsMax  = [cells[i].getText() for i in range( 5, len(cells), ncolumns)]
    Share   = [cells[i].getText() for i in range( 6, len(cells), ncolumns)]
    G       = [cells[i].getText() for i in range( 7, len(cells), ncolumns)]
    MP      = [cells[i].getText() for i in range( 8, len(cells), ncolumns)]
    PTS     = [cells[i].getText() for i in range( 9, len(cells), ncolumns)]
    TRB     = [cells[i].getText() for i in range(10, len(cells), ncolumns)]
    AST     = [cells[i].getText() for i in range(11, len(cells), ncolumns)]
    STL     = [cells[i].getText() for i in range(12, len(cells), ncolumns)]
    BLK     = [cells[i].getText() for i in range(13, len(cells), ncolumns)]
    FGP     = [cells[i].getText() for i in range(14, len(cells), ncolumns)]
    TPP     = [cells[i].getText() for i in range(15, len(cells), ncolumns)]
    FTP     = [cells[i].getText() for i in range(16, len(cells), ncolumns)]
    WS      = [cells[i].getText() for i in range(17, len(cells), ncolumns)]
    WS48    = [cells[i].getText() for i in range(18, len(cells), ncolumns)]

    Player = [i.replace('*', '') for i in Player] # Remove possible asterix from player name
    
    for i in range(0, int(len(cells) / ncolumns)):
        row = [Player[i], Age[i], Tm[i], First[i], PtsWon[i], PtsMax[i], Share[i],
               G[i], MP[i], PTS[i], TRB[i], AST[i], STL[i], BLK[i], FGP[i], TPP[i], FTP[i], WS[i], WS48[i]]
        csv_writer.writerow(row)

        
def NBA_SMOY_scraper(year):
    """
    NBA SMOY voting data scraper function
    """

    import requests
    import csv
    from bs4 import BeautifulSoup

    out_path = NBAanalysisdir + 'data/NBA_SMOY_{}-{}.csv'.format(year-1, year)
    csv_file = open(out_path, 'w')
    csv_writer = csv.writer(csv_file)

    features = ['Player', 'Age', 'Tm', 'First', 'PtsWon', 'PtsMax', 'Share',
                'G', 'MP', 'PTS', 'TRB', 'AST', 'STL', 'BLK', 'FG%', '3P%', 'FT%', 'WS', 'WS/48']
    
    csv_writer.writerow(features)
    
    URL = 'https://www.basketball-reference.com/awards/awards_{}.html'.format(year)

    print("--- Scraping SMOY data {}-{}...".format(year-1, year))
    
    r = requests.get(URL)
    soup = BeautifulSoup(r.text, "html5lib")

    table_text = soup.find(text=re.compile('table class="sortable stats_table" id="smoy"'))
    table_soup = BeautifulSoup(table_text, "html5lib")
    cells = table_soup.find_all('td')

    ncolumns = len(features)
    
    Player  = [cells[i].getText() for i in range( 0, len(cells), ncolumns)]
    Age     = [cells[i].getText() for i in range( 1, len(cells), ncolumns)]
    Tm      = [cells[i].getText() for i in range( 2, len(cells), ncolumns)]
    First   = [cells[i].getText() for i in range( 3, len(cells), ncolumns)]
    PtsWon  = [cells[i].getText() for i in range( 4, len(cells), ncolumns)]
    PtsMax  = [cells[i].getText() for i in range( 5, len(cells), ncolumns)]
    Share   = [cells[i].getText() for i in range( 6, len(cells), ncolumns)]
    G       = [cells[i].getText() for i in range( 7, len(cells), ncolumns)]
    MP      = [cells[i].getText() for i in range( 8, len(cells), ncolumns)]
    PTS     = [cells[i].getText() for i in range( 9, len(cells), ncolumns)]
    TRB     = [cells[i].getText() for i in range(10, len(cells), ncolumns)]
    AST     = [cells[i].getText() for i in range(11, len(cells), ncolumns)]
    STL     = [cells[i].getText() for i in range(12, len(cells), ncolumns)]
    BLK     = [cells[i].getText() for i in range(13, len(cells), ncolumns)]
    FGP     = [cells[i].getText() for i in range(14, len(cells), ncolumns)]
    TPP     = [cells[i].getText() for i in range(15, len(cells), ncolumns)]
    FTP     = [cells[i].getText() for i in range(16, len(cells), ncolumns)]
    WS      = [cells[i].getText() for i in range(17, len(cells), ncolumns)]
    WS48    = [cells[i].getText() for i in range(18, len(cells), ncolumns)]

    Player = [i.replace('*', '') for i in Player] # Remove possible asterix from player name
    
    for i in range(0, int(len(cells) / ncolumns)):
        row = [Player[i], Age[i], Tm[i], First[i], PtsWon[i], PtsMax[i], Share[i],
               G[i], MP[i], PTS[i], TRB[i], AST[i], STL[i], BLK[i], FGP[i], TPP[i], FTP[i], WS[i], WS48[i]]
        csv_writer.writerow(row)

        
def team_info(team_acronym):

    if (team_acronym == 'ATL'):
        full_team_name = 'Atlanta Hawks'
        conference = 'EC'
    elif (team_acronym == 'BOS'):
        full_team_name = 'Boston Celtics'
        conference = 'EC'
    elif (team_acronym == 'BRK'):
        full_team_name = 'Brooklyn Nets'
        conference = 'EC'
    elif (team_acronym == 'CHA'):
        full_team_name = 'Charlotte Bobcats'
        conference = 'EC'
    elif (team_acronym == 'CHO' or team_acronym == 'CHH'):
        full_team_name = 'Charlotte Hornets'
        conference = 'EC'
    elif (team_acronym == 'CHI'):
        full_team_name = 'Chicago Bulls'
        conference = 'EC'
    elif (team_acronym == 'CLE'):
        full_team_name = 'Cleveland Cavaliers'
        conference = 'EC'
    elif (team_acronym == 'DAL'):
        full_team_name = 'Dallas Mavericks'
        conference = 'WC'
    elif (team_acronym == 'DEN'):
        full_team_name = 'Denver Nuggets'
        conference = 'WC'
    elif (team_acronym == 'DET'):
        full_team_name = 'Detroit Pistons'
        conference = 'EC'
    elif (team_acronym == 'GSW'):
        full_team_name = 'Golden State Warriors'
        conference = 'WC'
    elif (team_acronym == 'HOU'):
        full_team_name = 'Houston Rockets'
        conference = 'WC'
    elif (team_acronym == 'IND'):
        full_team_name = 'Indiana Pacers'
        conference = 'EC'
    elif (team_acronym == 'KCK'):
        full_team_name = 'Kansas City Kings'
        conference = 'WC'
    elif (team_acronym == 'LAC'):
        full_team_name = 'Los Angeles Clippers'
        conference = 'WC'
    elif (team_acronym == 'LAL'):
        full_team_name = 'Los Angeles Lakers'
        conference = 'WC'
    elif (team_acronym == 'MEM'):
        full_team_name = 'Memphis Grizzlies'
        conference = 'WC'
    elif (team_acronym == 'MIA'):
        full_team_name = 'Miami Heat'
        conference = 'EC'
    elif (team_acronym == 'MIL'):
        full_team_name = 'Milwaukee Bucks'
        conference = 'EC'
    elif (team_acronym == 'MIN'):
        full_team_name = 'Minnesota Timberwolves'
        conference = 'WC'
    elif (team_acronym == 'NJN'):
        full_team_name = 'New Jersey Nets'
        conference = 'EC'
    elif (team_acronym == 'NOH'):
        full_team_name = 'New Orleans Hornets'
        conference = 'WC'
    elif (team_acronym == 'NOK'):
        full_team_name = 'New Orleans/Oklahoma City Hornets'
        conference = 'WC'
    elif (team_acronym == 'NOP'):
        full_team_name = 'New Orleans Pelicans'
        conference = 'WC'
    elif (team_acronym == 'NYK'):
        full_team_name = 'New York Knicks'
        conference = 'EC'
    elif (team_acronym == 'OKC'):
        full_team_name = 'Oklahoma City Thunder'
        conference = 'WC'
    elif (team_acronym == 'ORL'):
        full_team_name = 'Orlando Magic'
        conference = 'EC'
    elif (team_acronym == 'PHI'):
        full_team_name = 'Philadelphia 76ers'
        conference = 'EC'
    elif (team_acronym == 'PHO'):
        full_team_name = 'Phoenix Suns'
        conference = 'WC'
    elif (team_acronym == 'POR'):
        full_team_name = 'Portland Trail Blazers'
        conference = 'WC'
    elif (team_acronym == 'SAC'):
        full_team_name = 'Sacramento Kings'
        conference = 'WC'
    elif (team_acronym == 'SAS'):
        full_team_name = 'San Antonio Spurs'
        conference = 'WC'
    elif (team_acronym == 'SDC'):
        full_team_name = 'San Diego Clippers'
        conference = 'WC'
    elif (team_acronym == 'SEA'):
        full_team_name = 'Seattle SuperSonics'
        conference = 'WC'
    elif (team_acronym == 'TOR'):
        full_team_name = 'Toronto Raptors'
        conference = 'EC'
    elif (team_acronym == 'UTA'):
        full_team_name = 'Utah Jazz'
        conference = 'WC'
    elif (team_acronym == 'VAN'):
        full_team_name = 'Vancouver Grizzlies'
        conference = 'WC'
    elif (team_acronym == 'WAS'):
        full_team_name = 'Washington Wizards'
        conference = 'EC'
    elif (team_acronym == 'WSB'):
        full_team_name = 'Washington Bullets'
        conference = 'EC'
    else:
        print("")
        print("*** team_info ERROR ****")
        print("")
        exit()
        
    return full_team_name, conference


def logistic_function(x):
    return 1 / (1 + np.exp(-x))


