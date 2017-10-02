import glob
import math
import numpy as np
import pandas as pd

class footballdata():
    """description of class"""
    #pull data from stats folder method


    #[home(0) or away(1), Opponent(int), O1stDowns, OPassYards, ORunYards,OTOs,D1stDowns, DPassYards, DRunYards, DTOs]
    def getData(directory,teams,scores):
        i = 0
        for file in glob.iglob(directory):
            seasonfile = pd.read_csv(file)
            season = []
            seasonscores = []
            for index, row in seasonfile.iterrows():
                game = []
                #offense
                for x in range(8,17):
                    if x ==8:
                        #home(0) & away(1)
                        if row[x] == "@":
                            game.append(1)
                        else:
                            game.append(0)
                        continue
                    #team name
                    if isinstance(row[x],str):
                        game.append(assign_team_number(row[x]))
                        continue
                    #skip total yards
                    if x == 13:
                        continue
                    #check for value
                    if math.isnan(row[x]):
                        game.append(0)
                    else:
                        game.append(row[x]) 
                #defense
                for x in range(17,22):
                    #skip total yards
                    if x == 18:
                        continue
                    #check for value
                    if math.isnan(row[x]):
                        game.append(0)
                    else:
                        game.append(row[x])
                if game[1] == "Bye Week":
                    continue
                else:
                    season.append(game)
                    seasonscores.append(row[10])  
            teams[i].append(season)
            scores[i].append(seasonscores)
            i += 1

    ##Average Methods
    def create_average_data(teamStats,teamAverageStats):
        tempAverage = []
        for season in teamStats:
            for game in season:
                tempAverage.append(game[4:12])
                a = np.array(tempAverage)
                a = a.mean(axis=0)
                x = list(a)
                #rounding not sure if we need to
                x = [round(float(i),2) for i in x]
                teamAverageStats.append(x)

    def add_average_game(game, teamAverageStats):
        tempAverage = []
        for x in teamAverageStats:
            tempAverage.append(x)
        tempAverage.append(game)
        a = np.array(tempAverage)
        a = a.mean(axis=0)
        x = list(a)
        #rounding not sure if we need to
        x = [round(float(i),2) for i in x]
        teamAverageStats.append(x)

    def flatten_array(array1):
        newArray = []
        for season in array1:
            for game in season:
                newArray.append(game)
        return newArray

def assign_team_number(teamName):
    if teamName == "Bye Week":
        return "Bye Week"
    if teamName == "Arizona Cardinals":
        return 0
    if teamName == "Atlanta Falcons":
        return 1
    if teamName == "Baltimore Ravens":
        return 2
    if teamName == "Buffalo Bills":
        return 3
    if teamName == "Carolina Panthers":
        return 4
    if teamName == "Chicago Bears":
        return 5
    if teamName == "Cincinnati Bengals":
        return 6
    if teamName == "Cleveland Browns":
        return 7
    if teamName == "Dallas Cowboys":
        return 8
    if teamName == "Denver Broncos":
        return 9
    if teamName == "Detroit Lions":
        return 10
    if teamName == "Green Bay Packers":
        return 11
    if teamName == "Houston Texans":
        return 12
    if teamName == "Indianapolis Colts":
        return 13
    if teamName == "Jacksonville Jaguars":
        return 14
    if teamName == "Kansas City Chiefs":
        return 15
    if teamName == "Los Angeles Rams":
        return 16
    if teamName == "St. Louis Rams":
        return 16
    if teamName == "Miami Dolphins":
        return 17
    if teamName == "Minnesota Vikings":
        return 18
    if teamName == "New England Patriots":
        return 19
    if teamName == "New Orleans Saints":
        return 20
    if teamName == "New York Giants":
        return 21
    if teamName == "New York Jets":
        return 22
    if teamName == "Oakland Raiders":
        return 23
    if teamName == "Philadelphia Eagles":
        return 24
    if teamName == "Pittsburgh Steelers":
        return 25
    if teamName == "San Diego Chargers":
        return 26
    if teamName == "San Francisco 49ers":
        return 27
    if teamName == "Seattle Seahawks":
        return 28
    if teamName == "Tampa Bay Buccaneers":
        return 29
    if teamName == "Tennessee Titans":
        return 30
    if teamName == "Washington Redskins":
        return 31