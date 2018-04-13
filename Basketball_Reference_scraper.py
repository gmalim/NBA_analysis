#!/usr/local/bin/python3 -tt
"""
Program to scrape NBA player data from https://www.basketball-reference.com/.

Author: Gordon Lim
Last Edit: 13 Apr 2018 
"""

import NBAanalysissetup
import sys

def main():

    firstyear = 2010
    if (len(sys.argv) > 1):
        firstyear_str = sys.argv[1]
        firstyear = int(firstyear_str)    
    
    lastyear = 2017
    if (len(sys.argv) > 2):
        lastyear_str = sys.argv[2]
        lastyear = int(lastyear_str)    
    
    for year in range(firstyear, lastyear):
        #NBAanalysissetup.NBA_totals_scraper(year)
        #NBAanalysissetup.NBA_advanced_scraper(year)
        #NBAanalysissetup.NBA_AllStar_scraper(year)
        #NBAanalysissetup.NBA_teamstats_scraper(year)
        #NBAanalysissetup.NBA_teammisc_scraper(year)
        NBAanalysissetup.NBA_MVP_scraper(year)

    return 0

if __name__ == '__main__':
    main()
