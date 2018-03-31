#!/usr/local/bin/python3 -tt
"""
NBA analysis setup.

Change the following directory according to your system setup:
"""

myNBAanalysisdir  = "${HOME}/Programming/github_reps/NBA_analysis/" # Your NBA analysis directory

"""

Author: Gordon Lim
Last Edit: 23 Mar 2018
"""

import itertools
import matplotlib.pyplot as plt #; plt.style.use('ggplot')
import numpy as np
import os
import pandas as pd
import re

global NBAanalysisdir    

NBAanalysisdir = os.path.expandvars(myNBAanalysisdir)
    
if not os.path.isdir(NBAanalysisdir):
    print("--- ERROR: {} does not exist - EXIT".format(NBAanalysisdir))
    exit()

def loaddata_allyears(prediction_year, validation_year, training_years, includeadvancedstats):
    """
    Function that loads NBA data from csv-files for all years
    """
    
    # Load training year data into df_training:

    dfs = []

    for training_year in training_years:
        
        print("--> Loading   training year {}-{} ...".format(training_year-1, training_year))
        df = loaddata_singleyear(training_year, includeadvancedstats)
        dfs.append(df)

    df_training = pd.concat(dfs)
    #print(df_training.head())
    #print(df_training.shape)
    
    # Load validation data into df_validation:

    print("--> Loading validation year {}-{} ...".format(validation_year-1, validation_year))
    df_validation = loaddata_singleyear(validation_year, includeadvancedstats)
    #print(df_validation.head())
    #print(df_validation.shape)

    # Load prediction data into df_prediction:

    print("--> Loading prediction year {}-{} ...".format(prediction_year-1, prediction_year))
    df_prediction = loaddata_singleyear(prediction_year, includeadvancedstats)
    #print(df_prediction.head())
    #print(df_prediction.shape)
    
    return df_training, df_validation, df_prediction


def loaddata_singleyear(year, includeadvancedstats):
    """
    Function that loads NBA data from csv-files for one particular year
    """

    NBA_playerstats_csvfilename = NBAanalysisdir + 'data/NBA_totals_{}-{}.csv'.format(year-1, year)
    
    if not os.path.isfile(NBA_playerstats_csvfilename):
        print("--- ERROR: {} does not exist - EXIT".format(NBA_playerstats_csvfilename))
        exit()
    
    df = pd.read_csv(NBA_playerstats_csvfilename)

    #print(df.head())    
    #print(df.shape)

    if includeadvancedstats:
    
        NBA_playerstats_advanced_csvfilename = NBAanalysisdir + 'data/NBA_advanced_{}-{}.csv'.format(year-1, year)
    
        if not os.path.isfile(NBA_playerstats_advanced_csvfilename):
            print("--- ERROR: {} does not exist - EXIT")
            exit()
    
        df2 = pd.read_csv(NBA_playerstats_advanced_csvfilename)
        #df2.drop(df2.columns[[19, 24]], inplace=True, axis=1)    # remove empty columns
        df2.drop(['Pos', 'Age', 'G', 'MP'], inplace=True, axis=1) # remove columns already included in regular stats csv

        #print(df2.head())    
        #print(df2.shape)

        df = pd.merge(df, df2, how='left', left_on=['Player', 'Tm'], right_on=['Player', 'Tm'])

    #print(df.head())    
    #print(df.shape)

    # Clean player names:

    df = cleanplayernames(df)

    # For players with more than one row, keep only row with 'Tm' == 'TOT' and replace 'Tm' value with most recent team id:

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

    # Add All-Star statistics:
                
    df = add_AllStar_column(year, df)

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

    df_as['AS'] = 1
    
    #df = pd.merge(df, df_as, how='left', left_on=['Player'], right_on=['Starters'])
    df = pd.merge(df, df_as, how='left', left_on=['Player'], right_on=['Player'])
    #df = df.drop(['Starters'], axis=1) 
    
    values = {'AS': 0}
    df.fillna(value=values, inplace=True) # replace NaNs with 0s

    df['AS'] = df['AS'].astype('int64')
    
    return df


def cleanplayernames(df):
    """
    #This function cleans up player names by removing the part that starts with "\":
    This function cleans up player names by removing a trailing asterix from a name if present:
    """

    #df['Player'].replace(to_replace=r'(\\[\w\d]+$)', value='', regex=True, inplace=True)
    df['Player'].replace(to_replace=r'(\*$)', value='', regex=True, inplace=True)

    return df


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
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
            
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
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

    Player  = [cells[i].getText() for i in range( 0, len(cells), 29)]
    Pos     = [cells[i].getText() for i in range( 1, len(cells), 29)]
    Age     = [cells[i].getText() for i in range( 2, len(cells), 29)]
    Tm      = [cells[i].getText() for i in range( 3, len(cells), 29)]
    G       = [cells[i].getText() for i in range( 4, len(cells), 29)]
    GS      = [cells[i].getText() for i in range( 5, len(cells), 29)]
    MP      = [cells[i].getText() for i in range( 6, len(cells), 29)]
    FG      = [cells[i].getText() for i in range( 7, len(cells), 29)]
    FGA     = [cells[i].getText() for i in range( 8, len(cells), 29)]
    FGP     = [cells[i].getText() for i in range( 9, len(cells), 29)]
    THP     = [cells[i].getText() for i in range(10, len(cells), 29)]
    THPA    = [cells[i].getText() for i in range(11, len(cells), 29)]
    THPP    = [cells[i].getText() for i in range(12, len(cells), 29)]
    TWP     = [cells[i].getText() for i in range(13, len(cells), 29)]
    TWPA    = [cells[i].getText() for i in range(14, len(cells), 29)]
    TWPP    = [cells[i].getText() for i in range(15, len(cells), 29)]
    EFGP    = [cells[i].getText() for i in range(16, len(cells), 29)]
    FT      = [cells[i].getText() for i in range(17, len(cells), 29)]
    FTA     = [cells[i].getText() for i in range(18, len(cells), 29)]
    FTP     = [cells[i].getText() for i in range(19, len(cells), 29)]
    ORB     = [cells[i].getText() for i in range(20, len(cells), 29)]
    DRB     = [cells[i].getText() for i in range(21, len(cells), 29)]
    TRB     = [cells[i].getText() for i in range(22, len(cells), 29)]
    AST     = [cells[i].getText() for i in range(23, len(cells), 29)]
    STL     = [cells[i].getText() for i in range(24, len(cells), 29)]
    BLK     = [cells[i].getText() for i in range(25, len(cells), 29)]
    TOV     = [cells[i].getText() for i in range(26, len(cells), 29)]
    PF      = [cells[i].getText() for i in range(27, len(cells), 29)]
    PTS     = [cells[i].getText() for i in range(28, len(cells), 29)]
    
    for i in range(0, int(len(cells) / 29)):
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

    Player  = [cells[i].getText() for i in range( 0, len(cells), 28)]
    Pos     = [cells[i].getText() for i in range( 1, len(cells), 28)]
    Age     = [cells[i].getText() for i in range( 2, len(cells), 28)]
    Tm      = [cells[i].getText() for i in range( 3, len(cells), 28)]
    G       = [cells[i].getText() for i in range( 4, len(cells), 28)]
    MP      = [cells[i].getText() for i in range( 5, len(cells), 28)]
    PER     = [cells[i].getText() for i in range( 6, len(cells), 28)]
    TSP     = [cells[i].getText() for i in range( 7, len(cells), 28)]
    TPAr    = [cells[i].getText() for i in range( 8, len(cells), 28)]
    FTr     = [cells[i].getText() for i in range( 9, len(cells), 28)]
    ORBP    = [cells[i].getText() for i in range(10, len(cells), 28)]
    DRBP    = [cells[i].getText() for i in range(11, len(cells), 28)]
    TRBP    = [cells[i].getText() for i in range(12, len(cells), 28)]
    ASTP    = [cells[i].getText() for i in range(13, len(cells), 28)]
    STLP    = [cells[i].getText() for i in range(14, len(cells), 28)]
    BLKP    = [cells[i].getText() for i in range(15, len(cells), 28)]
    TOVP    = [cells[i].getText() for i in range(16, len(cells), 28)]
    USGP    = [cells[i].getText() for i in range(17, len(cells), 28)]
    OWS     = [cells[i].getText() for i in range(19, len(cells), 28)] # 18 is empty!
    DWS     = [cells[i].getText() for i in range(20, len(cells), 28)]
    WS      = [cells[i].getText() for i in range(21, len(cells), 28)]
    WS48    = [cells[i].getText() for i in range(22, len(cells), 28)]
    OBPM    = [cells[i].getText() for i in range(24, len(cells), 28)] # 23 is empty!
    DBPM    = [cells[i].getText() for i in range(25, len(cells), 28)]
    BPM     = [cells[i].getText() for i in range(26, len(cells), 28)]
    VORP    = [cells[i].getText() for i in range(27, len(cells), 28)]
    
    for i in range(0, int(len(cells) / 28)):
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
        
    Tm  = [cells[i].getText() for i in range( 0, len(cells), 20)]
    MP  = [cells[i].getText() for i in range( 1, len(cells), 20)]
    FG  = [cells[i].getText() for i in range( 2, len(cells), 20)]
    FGA = [cells[i].getText() for i in range( 3, len(cells), 20)]
    TP  = [cells[i].getText() for i in range( 4, len(cells), 20)]
    TPA = [cells[i].getText() for i in range( 5, len(cells), 20)]
    FT  = [cells[i].getText() for i in range( 6, len(cells), 20)]
    FTA = [cells[i].getText() for i in range( 7, len(cells), 20)]
    ORB = [cells[i].getText() for i in range( 8, len(cells), 20)]
    DRB = [cells[i].getText() for i in range( 9, len(cells), 20)]
    TRB = [cells[i].getText() for i in range(10, len(cells), 20)]
    AST = [cells[i].getText() for i in range(11, len(cells), 20)]
    STL = [cells[i].getText() for i in range(12, len(cells), 20)]
    BLK = [cells[i].getText() for i in range(13, len(cells), 20)]
    TOV = [cells[i].getText() for i in range(14, len(cells), 20)]
    PF  = [cells[i].getText() for i in range(15, len(cells), 20)]
    PTS = [cells[i].getText() for i in range(16, len(cells), 20)]
    FGP = [cells[i].getText() for i in range(17, len(cells), 20)]
    TPP = [cells[i].getText() for i in range(18, len(cells), 20)]
    FTP = [cells[i].getText() for i in range(19, len(cells), 20)]
    
    for i in range(0, int(len(cells) / 20) - 1):
        row = [players[i].getText(), Tm[i], MP[i], FG[i], FGA[i], TP[i], TPA[i], FT[i], FTA[i], ORB[i], \
               DRB[i], TRB[i], AST[i], STL[i], BLK[i], TOV[i], PF[i], PTS[i], FGP[i], TPP[i], FTP[i]]
        csv_writer.writerow(row)

    return 0
