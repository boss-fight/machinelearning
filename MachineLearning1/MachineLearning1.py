from sklearn import tree
from sklearn.svm import SVR
from sklearn import preprocessing
import numpy as np
#import pandas as pd
import csv
import math
import glob

from footballdata import footballdata

#CSV files -> https://www.pro-football-reference.com/

#Panda Data gather
#pd.set_option('display.max_rows',500)
#pd.set_option('display.max_columns', 50)
#pd.set_option('display.width', 1000)

gameHeaders = ["team score", "opp score", "O1stD", "OTotYd", "OPassY", "ORushY", "TO",
              "D1stD", "DTotYd", "DPassY", "DRushY", "DTO","Expected Offense Points",
              "Expected Defense Points", "Expected Special Points"]


##methods
def season_walkthrough(teamStats,teamScores, teamStats2016, teamScores2016, svr):
    #original trainData from above method minus range parameter
    newTeamStats = []
    for season in teamStats:
        for game in season:
            newTeamStats.append(game)
    newerTeamStats = flatten_array(teamStats)
    newTeamScores = []
    for season in teamScores:
        for game in season:
            newTeamScores.append(game)
 
    newerTeamScores = flatten_array(teamScores)
    #first fit
    svr.fit(newerTeamStats,newerTeamScores)

    #pull out 2016 stats/scores
    newTeamStats2016 = []
    newTeamScores2016 = []
    for season in teamStats2016:
        for game in season:
            newTeamStats2016.append(game)

    for season in teamScores2016:
        for game in season:
            newTeamScores2016.append(game)

    #iterate through 2016 and print predictions
    while (len(newTeamStats2016)>0):
        game = newTeamStats2016.pop(0)
        score = newTeamScores2016.pop(0)
        
        #add data to old before reshape
        newerTeamStats.append(game)
        newerTeamScores.append(score)
        
        #predict
        game = np.reshape(game,(1,-1))
        print("predicted score")
        print(svr.predict(game))
        print("actual score")
        print(score)
        svr.fit(newerTeamStats,newerTeamScores)

def train_average_points(teamstats,teamscores):
    print("yikes")


def grab_average_offense(averageTeamStats):
    team1AverageOffense = []
    for game in averageTeamStats:
        team1AverageOffense.append(game[0:4])
    return team1AverageOffense

def grab_average_offense_game(averageTeamGame):
    defense = averageTeamGame[0:4]
    return defense

def grab_average_defense(averageTeamStats):
    team2AverageDefense = []
    for game in averageTeamStats:
        team2AverageDefense.append(game[4:8])
    return team2AverageDefense

def grab_average_defense_game(averageTeamGame):
    defense = averageTeamGame[4:8]
    return defense

def create_offense_feature_data(regularTeamStats,averageTeamStats, allTeamAverages):
    featureSet = []
    averageOffense = grab_average_offense(averageTeamStats)
    flatRegular = footballdata.flatten_array(regularTeamStats)
    i = 0
    for game in averageTeamStats:
        teamNumber = int(flatRegular[i][1])
        dstats = grab_average_defense_game(allTeamAverages[teamNumber][i])
        featureData = [a - b for a, b in zip(averageOffense[i],dstats)]
        featureSet.append(featureData)
        i += 1
    #remove first 16 games
    featureSet = featureSet[32:len(featureSet)]
    return featureSet

def predict_offense_firstdowns(featureSet,teamStats):
    flatStats = footballdata.flatten_array(teamStats)
    actualFirstdowns = []
    i = 0
    for game in featureSet:
        actualFirstdowns.append(flatStats[i][4])
        i += 1
    svr = SVR(kernel='linear', C=1e3)
    svr.fit(featureSet,actualFirstdowns)
    return svr

def predict_offense_passyards(featureSet,teamStats):
    flatStats = footballdata.flatten_array(teamStats)
    actualyards = []
    i = 0
    for game in featureSet:
        actualyards.append(flatStats[i][5])
        i += 1
    svr = SVR(kernel='linear', C=1e3)
    svr.fit(featureSet,actualyards)
    return svr

def predict_offense_rushyards(featureSet,teamStats):
    flatStats = footballdata.flatten_array(teamStats)
    actualyards = []
    i = 0
    for game in featureSet:
        actualyards.append(flatStats[i][6])
        i += 1
    svr = SVR(kernel='linear', C=1e3)
    svr.fit(featureSet,actualyards)
    return svr

def predict_offense_turnovers(featureSet,teamStats):
    flatStats = footballdata.flatten_array(teamStats)
    actualyards = []
    i = 0
    for game in featureSet:
        actualyards.append(flatStats[i][7])
        i += 1
    svr = SVR(kernel='linear', C=1e3)
    svr.fit(featureSet,actualyards)
    return svr


def create_defense_feature_data(regularTeamStats,averageTeamStats,allTeamAverages):
    print("yikes")


#2012 stats
arizona2012stats = []
atlanta2012stats = []
baltimore2012stats = []
buffalo2012stats = []
carolina2012stats = []
chicago2012stats = []
cincinnati2012stats = []
cleveland2012stats = []
dallas2012stats = []
denver2012stats = []
detroit2012stats = []
greenbay2012stats = []
houston2012stats = []
indianapolis2012stats = []
jacksonville2012stats = []
kansascity2012stats = []
losangeles2012stats = []
miami2012stats = []
minnesota2012stats = []
newengland2012stats = []
neworleans2012stats = []
newyorkg2012stats = []
newyorkj2012stats = []
oakland2012stats = []
philadelphia2012stats = []
pittsburgh2012stats = []
sandiego2012stats = []
sanfran2012stats = []
seattle2012stats = []
tampabay2012stats = []
tennessee2012stats = []
washington2012stats = []
#2012 scores
arizona2012scores = []
atlanta2012scores = []
baltimore2012scores = []
buffalo2012scores = []
carolina2012scores = []
chicago2012scores = []
cincinnati2012scores = []
cleveland2012scores = []
dallas2012scores = []
denver2012scores = []
detroit2012scores = []
greenbay2012scores = []
houston2012scores = []
indianapolis2012scores = []
jacksonville2012scores = []
kansascity2012scores = []
losangeles2012scores = []
miami2012scores = []
minnesota2012scores = []
newengland2012scores = []
neworleans2012scores = []
newyorkg2012scores = []
newyorkj2012scores = []
oakland2012scores = []
philadelphia2012scores = []
pittsburgh2012scores = []
sandiego2012scores = []
sanfran2012scores = []
seattle2012scores = []
tampabay2012scores = []
tennessee2012scores = []
washington2012scores = []

directory2012 = './footballdata/2012/*.txt'

teams2012 = [arizona2012stats,atlanta2012stats,baltimore2012stats,buffalo2012stats,carolina2012stats,chicago2012stats,cincinnati2012stats,cleveland2012stats,dallas2012stats,denver2012stats,
                detroit2012stats,greenbay2012stats,houston2012stats,indianapolis2012stats,jacksonville2012stats,kansascity2012stats,losangeles2012stats,miami2012stats,minnesota2012stats,
                newengland2012stats,neworleans2012stats,newyorkg2012stats,newyorkj2012stats,oakland2012stats,philadelphia2012stats,pittsburgh2012stats,sandiego2012stats,sanfran2012stats,
            seattle2012stats,tampabay2012stats,tennessee2012stats,washington2012stats]


scores2012 = [arizona2012scores ,atlanta2012scores,baltimore2012scores,buffalo2012scores,carolina2012scores,chicago2012scores,cincinnati2012scores,cleveland2012scores,dallas2012scores,denver2012scores,
                detroit2012scores,greenbay2012scores,houston2012scores,indianapolis2012scores,jacksonville2012scores,kansascity2012scores,losangeles2012scores,miami2012scores,minnesota2012scores,
                newengland2012scores,neworleans2012scores,newyorkg2012scores,newyorkj2012scores,oakland2012scores,philadelphia2012scores,pittsburgh2012scores,sandiego2012scores,sanfran2012scores,
                seattle2012scores,tampabay2012scores,tennessee2012scores,washington2012scores]

#2013 stats
arizona2013stats = []
atlanta2013stats = []
baltimore2013stats = []
buffalo2013stats = []
carolina2013stats = []
chicago2013stats = []
cincinnati2013stats = []
cleveland2013stats = []
dallas2013stats = []
denver2013stats = []
detroit2013stats = []
greenbay2013stats = []
houston2013stats = []
indianapolis2013stats = []
jacksonville2013stats = []
kansascity2013stats = []
losangeles2013stats = []
miami2013stats = []
minnesota2013stats = []
newengland2013stats = []
neworleans2013stats = []
newyorkg2013stats = []
newyorkj2013stats = []
oakland2013stats = []
philadelphia2013stats = []
pittsburgh2013stats = []
sandiego2013stats = []
sanfran2013stats = []
seattle2013stats = []
tampabay2013stats = []
tennessee2013stats = []
washington2013stats = []
#2013 scores
arizona2013scores = []
atlanta2013scores = []
baltimore2013scores = []
buffalo2013scores = []
carolina2013scores = []
chicago2013scores = []
cincinnati2013scores = []
cleveland2013scores = []
dallas2013scores = []
denver2013scores = []
detroit2013scores = []
greenbay2013scores = []
houston2013scores = []
indianapolis2013scores = []
jacksonville2013scores = []
kansascity2013scores = []
losangeles2013scores = []
miami2013scores = []
minnesota2013scores = []
newengland2013scores = []
neworleans2013scores = []
newyorkg2013scores = []
newyorkj2013scores = []
oakland2013scores = []
philadelphia2013scores = []
pittsburgh2013scores = []
sandiego2013scores = []
sanfran2013scores = []
seattle2013scores = []
tampabay2013scores = []
tennessee2013scores = []
washington2013scores = []

directory2013 = './footballdata/2013/*.txt'

teams2013 = [arizona2013stats,atlanta2013stats,baltimore2013stats,buffalo2013stats,carolina2013stats,chicago2013stats,cincinnati2013stats,cleveland2013stats,dallas2013stats,denver2013stats,
                detroit2013stats,greenbay2013stats,houston2013stats,indianapolis2013stats,jacksonville2013stats,kansascity2013stats,losangeles2013stats,miami2013stats,minnesota2013stats,
                newengland2013stats,neworleans2013stats,newyorkg2013stats,newyorkj2013stats,oakland2013stats,philadelphia2013stats,pittsburgh2013stats,sandiego2013stats,sanfran2013stats,
            seattle2013stats,tampabay2013stats,tennessee2013stats,washington2013stats]


scores2013 = [arizona2013scores ,atlanta2013scores,baltimore2013scores,buffalo2013scores,carolina2013scores,chicago2013scores,cincinnati2013scores,cleveland2013scores,dallas2013scores,denver2013scores,
                detroit2013scores,greenbay2013scores,houston2013scores,indianapolis2013scores,jacksonville2013scores,kansascity2013scores,losangeles2013scores,miami2013scores,minnesota2013scores,
                newengland2013scores,neworleans2013scores,newyorkg2013scores,newyorkj2013scores,oakland2013scores,philadelphia2013scores,pittsburgh2013scores,sandiego2013scores,sanfran2013scores,
                seattle2013scores,tampabay2013scores,tennessee2013scores,washington2013scores]

#2014 stats
arizona2014stats = []
atlanta2014stats = []
baltimore2014stats = []
buffalo2014stats = []
carolina2014stats = []
chicago2014stats = []
cincinnati2014stats = []
cleveland2014stats = []
dallas2014stats = []
denver2014stats = []
detroit2014stats = []
greenbay2014stats = []
houston2014stats = []
indianapolis2014stats = []
jacksonville2014stats = []
kansascity2014stats = []
losangeles2014stats = []
miami2014stats = []
minnesota2014stats = []
newengland2014stats = []
neworleans2014stats = []
newyorkg2014stats = []
newyorkj2014stats = []
oakland2014stats = []
philadelphia2014stats = []
pittsburgh2014stats = []
sandiego2014stats = []
sanfran2014stats = []
seattle2014stats = []
tampabay2014stats = []
tennessee2014stats = []
washington2014stats = []
#2014 scores
arizona2014scores = []
atlanta2014scores = []
baltimore2014scores = []
buffalo2014scores = []
carolina2014scores = []
chicago2014scores = []
cincinnati2014scores = []
cleveland2014scores = []
dallas2014scores = []
denver2014scores = []
detroit2014scores = []
greenbay2014scores = []
houston2014scores = []
indianapolis2014scores = []
jacksonville2014scores = []
kansascity2014scores = []
losangeles2014scores = []
miami2014scores = []
minnesota2014scores = []
newengland2014scores = []
neworleans2014scores = []
newyorkg2014scores = []
newyorkj2014scores = []
oakland2014scores = []
philadelphia2014scores = []
pittsburgh2014scores = []
sandiego2014scores = []
sanfran2014scores = []
seattle2014scores = []
tampabay2014scores = []
tennessee2014scores = []
washington2014scores = []

directory2014 = './footballdata/2014/*.txt'

teams2014 = [arizona2014stats,atlanta2014stats,baltimore2014stats,buffalo2014stats,carolina2014stats,chicago2014stats,cincinnati2014stats,cleveland2014stats,dallas2014stats,denver2014stats,
                detroit2014stats,greenbay2014stats,houston2014stats,indianapolis2014stats,jacksonville2014stats,kansascity2014stats,losangeles2014stats,miami2014stats,minnesota2014stats,
                newengland2014stats,neworleans2014stats,newyorkg2014stats,newyorkj2014stats,oakland2014stats,philadelphia2014stats,pittsburgh2014stats,sandiego2014stats,sanfran2014stats,
            seattle2014stats,tampabay2014stats,tennessee2014stats,washington2014stats]


scores2014 = [arizona2014scores ,atlanta2014scores,baltimore2014scores,buffalo2014scores,carolina2014scores,chicago2014scores,cincinnati2014scores,cleveland2014scores,dallas2014scores,denver2014scores,
                detroit2014scores,greenbay2014scores,houston2014scores,indianapolis2014scores,jacksonville2014scores,kansascity2014scores,losangeles2014scores,miami2014scores,minnesota2014scores,
                newengland2014scores,neworleans2014scores,newyorkg2014scores,newyorkj2014scores,oakland2014scores,philadelphia2014scores,pittsburgh2014scores,sandiego2014scores,sanfran2014scores,
                seattle2014scores,tampabay2014scores,tennessee2014scores,washington2014scores]

#2015 stats
arizona2015stats = []
atlanta2015stats = []
baltimore2015stats = []
buffalo2015stats = []
carolina2015stats = []
chicago2015stats = []
cincinnati2015stats = []
cleveland2015stats = []
dallas2015stats = []
denver2015stats = []
detroit2015stats = []
greenbay2015stats = []
houston2015stats = []
indianapolis2015stats = []
jacksonville2015stats = []
kansascity2015stats = []
losangeles2015stats = []
miami2015stats = []
minnesota2015stats = []
newengland2015stats = []
neworleans2015stats = []
newyorkg2015stats = []
newyorkj2015stats = []
oakland2015stats = []
philadelphia2015stats = []
pittsburgh2015stats = []
sandiego2015stats = []
sanfran2015stats = []
seattle2015stats = []
tampabay2015stats = []
tennessee2015stats = []
washington2015stats = []
#2015 scores
arizona2015scores = []
atlanta2015scores = []
baltimore2015scores = []
buffalo2015scores = []
carolina2015scores = []
chicago2015scores = []
cincinnati2015scores = []
cleveland2015scores = []
dallas2015scores = []
denver2015scores = []
detroit2015scores = []
greenbay2015scores = []
houston2015scores = []
indianapolis2015scores = []
jacksonville2015scores = []
kansascity2015scores = []
losangeles2015scores = []
miami2015scores = []
minnesota2015scores = []
newengland2015scores = []
neworleans2015scores = []
newyorkg2015scores = []
newyorkj2015scores = []
oakland2015scores = []
philadelphia2015scores = []
pittsburgh2015scores = []
sandiego2015scores = []
sanfran2015scores = []
seattle2015scores = []
tampabay2015scores = []
tennessee2015scores = []
washington2015scores = []

directory2015 = './footballdata/2015/*.txt'

teams2015 = [arizona2015stats,atlanta2015stats,baltimore2015stats,buffalo2015stats,carolina2015stats,chicago2015stats,cincinnati2015stats,cleveland2015stats,dallas2015stats,denver2015stats,
                detroit2015stats,greenbay2015stats,houston2015stats,indianapolis2015stats,jacksonville2015stats,kansascity2015stats,losangeles2015stats,miami2015stats,minnesota2015stats,
                newengland2015stats,neworleans2015stats,newyorkg2015stats,newyorkj2015stats,oakland2015stats,philadelphia2015stats,pittsburgh2015stats,sandiego2015stats,sanfran2015stats,
            seattle2015stats,tampabay2015stats,tennessee2015stats,washington2015stats]


scores2015 = [arizona2015scores ,atlanta2015scores,baltimore2015scores,buffalo2015scores,carolina2015scores,chicago2015scores,cincinnati2015scores,cleveland2015scores,dallas2015scores,denver2015scores,
                detroit2015scores,greenbay2015scores,houston2015scores,indianapolis2015scores,jacksonville2015scores,kansascity2015scores,losangeles2015scores,miami2015scores,minnesota2015scores,
                newengland2015scores,neworleans2015scores,newyorkg2015scores,newyorkj2015scores,oakland2015scores,philadelphia2015scores,pittsburgh2015scores,sandiego2015scores,sanfran2015scores,
                seattle2015scores,tampabay2015scores,tennessee2015scores,washington2015scores]

#2016 data
#2016 stats
arizona2016stats = []
atlanta2016stats = []
baltimore2016stats = []
buffalo2016stats = []
carolina2016stats = []
chicago2016stats = []
cincinnati2016stats = []
cleveland2016stats = []
dallas2016stats = []
denver2016stats = []
detroit2016stats = []
greenbay2016stats = []
houston2016stats = []
indianapolis2016stats = []
jacksonville2016stats = []
kansascity2016stats = []
losangeles2016stats = []
miami2016stats = []
minnesota2016stats = []
newengland2016stats = []
neworleans2016stats = []
newyorkg2016stats = []
newyorkj2016stats = []
oakland2016stats = []
philadelphia2016stats = []
pittsburgh2016stats = []
sandiego2016stats = []
sanfran2016stats = []
seattle2016stats = []
tampabay2016stats = []
tennessee2016stats = []
washington2016stats = []
#2016 scores
arizona2016scores = []
atlanta2016scores = []
baltimore2016scores = []
buffalo2016scores = []
carolina2016scores = []
chicago2016scores = []
cincinnati2016scores = []
cleveland2016scores = []
dallas2016scores = []
denver2016scores = []
detroit2016scores = []
greenbay2016scores = []
houston2016scores = []
indianapolis2016scores = []
jacksonville2016scores = []
kansascity2016scores = []
losangeles2016scores = []
miami2016scores = []
minnesota2016scores = []
newengland2016scores = []
neworleans2016scores = []
newyorkg2016scores = []
newyorkj2016scores = []
oakland2016scores = []
philadelphia2016scores = []
pittsburgh2016scores = []
sandiego2016scores = []
sanfran2016scores = []
seattle2016scores = []
tampabay2016scores = []
tennessee2016scores = []
washington2016scores = []

directory2016 = './footballdata/2016/*.txt'

teams2016 = [arizona2016stats,atlanta2016stats,baltimore2016stats,buffalo2016stats,carolina2016stats,chicago2016stats,cincinnati2016stats,cleveland2016stats,dallas2016stats,denver2016stats,
                detroit2016stats,greenbay2016stats,houston2016stats,indianapolis2016stats,jacksonville2016stats,kansascity2016stats,losangeles2016stats,miami2016stats,minnesota2016stats,
                newengland2016stats,neworleans2016stats,newyorkg2016stats,newyorkj2016stats,oakland2016stats,philadelphia2016stats,pittsburgh2016stats,sandiego2016stats,sanfran2016stats,
            seattle2016stats,tampabay2016stats,tennessee2016stats,washington2016stats]


scores2016 = [arizona2016scores ,atlanta2016scores,baltimore2016scores,buffalo2016scores,carolina2016scores,chicago2016scores,cincinnati2016scores,cleveland2016scores,dallas2016scores,denver2016scores,
                detroit2016scores,greenbay2016scores,houston2016scores,indianapolis2016scores,jacksonville2016scores,kansascity2016scores,losangeles2016scores,miami2016scores,minnesota2016scores,
                newengland2016scores,neworleans2016scores,newyorkg2016scores,newyorkj2016scores,oakland2016scores,philadelphia2016scores,pittsburgh2016scores,sandiego2016scores,sanfran2016scores,
                seattle2016scores,tampabay2016scores,tennessee2016scores,washington2016scores]

#team averages
arizonaAverage = []
atlantaAverage = []
baltimoreAverage = []
buffaloAverage = []
carolinaAverage = []
chicagoAverage = []
cincinnatiAverage = []
clevelandAverage = []
dallasAverage = []
denverAverage = []
detroitAverage = []
greenbayAverage = []
houstonAverage = []
indianapolisAverage = []
jacksonvilleAverage = []
kansascityAverage = []
losangelesAverage = []
miamiAverage = []
minnesotaAverage = []
newenglandAverage = []
neworleansAverage = []
newyorkgAverage = []
newyorkjAverage = []
oaklandAverage = []
philadelphiaAverage = []
pittsburghAverage = []
sandiegoAverage = []
sanfranAverage = []
seattleAverage = []
tampabayAverage = []
tennesseeAverage = []
washingtonAverage = []

teamsAverages = [arizonaAverage,atlantaAverage,baltimoreAverage,buffaloAverage,carolinaAverage,chicagoAverage,cincinnatiAverage,clevelandAverage,dallasAverage,denverAverage,
                detroitAverage,greenbayAverage,houstonAverage,indianapolisAverage,jacksonvilleAverage,kansascityAverage,losangelesAverage,miamiAverage,minnesotaAverage,
                newenglandAverage,neworleansAverage,newyorkgAverage,newyorkjAverage,oaklandAverage,philadelphiaAverage,pittsburghAverage,sandiegoAverage,sanfranAverage,
            seattleAverage,tampabayAverage,tennesseeAverage,washingtonAverage]


#create team arrays    
footballdata.getData(directory2012,teams2012, scores2012)
footballdata.getData(directory2013,teams2013, scores2013)
footballdata.getData(directory2014,teams2014, scores2014)
footballdata.getData(directory2015,teams2015, scores2015)
footballdata.getData(directory2016,teams2016, scores2016)

#average seasons
arizonaAverageStats = arizona2012stats + arizona2013stats + arizona2014stats + arizona2015stats
arizonaAverageScores = arizona2012scores+ arizona2013scores + arizona2014scores + arizona2015scores
footballdata.create_average_data(arizonaAverageStats,arizonaAverage)
arizonaAverageScores = footballdata.flatten_array(arizonaAverageScores)

atlantaAverageStats = atlanta2012stats + atlanta2013stats + atlanta2014stats + atlanta2015stats
atlantaAverageScores = atlanta2012scores+ atlanta2013scores + atlanta2014scores + atlanta2015scores
footballdata.create_average_data(atlantaAverageStats,atlantaAverage)
atlantaAverageScores = footballdata.flatten_array(atlantaAverageScores)

baltimoreAverageStats = baltimore2012stats + baltimore2013stats + baltimore2014stats + baltimore2015stats
baltimoreAverageScores = baltimore2012scores+ baltimore2013scores + baltimore2014scores + baltimore2015scores
footballdata.create_average_data(baltimoreAverageStats,baltimoreAverage)
baltimoreAverageScores = footballdata.flatten_array(baltimoreAverageScores)

buffaloAverageStats = buffalo2012stats + buffalo2013stats + buffalo2014stats + buffalo2015stats
buffaloAverageScores = buffalo2012scores+ buffalo2013scores + buffalo2014scores + buffalo2015scores
footballdata.create_average_data(buffaloAverageStats,buffaloAverage)
buffaloAverageScores = footballdata.flatten_array(buffaloAverageScores)

carolinaAverageStats = carolina2012stats + carolina2013stats + carolina2014stats + carolina2015stats
carolinaAverageScores = carolina2012scores+ carolina2013scores + carolina2014scores + carolina2015scores
footballdata.create_average_data(arizonaAverageStats,carolinaAverage)
carolinaAverageScores = footballdata.flatten_array(carolinaAverageScores)

chicagoAverageStats = chicago2012stats + chicago2013stats + chicago2014stats + chicago2015stats
chicagoAverageScores = chicago2012scores+ chicago2013scores + chicago2014scores + chicago2015scores
footballdata.create_average_data(chicagoAverageStats,chicagoAverage)
chicagoAverageScores = footballdata.flatten_array(chicagoAverageScores)

cincinnatiAverageStats = cincinnati2012stats + cincinnati2013stats + cincinnati2014stats + cincinnati2015stats
cincinnatiAverageScores = cincinnati2012scores+ cincinnati2013scores + cincinnati2014scores + cincinnati2015scores
footballdata.create_average_data(cincinnatiAverageStats,cincinnatiAverage)
cincinnatiAverageScores = footballdata.flatten_array(cincinnatiAverageScores)

clevelandAverageStats = cleveland2012stats + cleveland2013stats + cleveland2014stats + cleveland2015stats
clevelandAverageScores = cleveland2012scores+ cleveland2013scores + cleveland2014scores + cleveland2015scores
footballdata.create_average_data(clevelandAverageStats,clevelandAverage)
clevelandAverageScores = footballdata.flatten_array(clevelandAverageScores)

dallasAverageStats = dallas2012stats + dallas2013stats + dallas2014stats + dallas2015stats
dallasAverageScores = dallas2012scores+ dallas2013scores + dallas2014scores + dallas2015scores
footballdata.create_average_data(dallasAverageStats,dallasAverage)
dallasAverageScores = footballdata.flatten_array(dallasAverageScores)

denverAverageStats = denver2012stats + denver2013stats + denver2014stats + denver2015stats
denverAverageScores = denver2012scores+ denver2013scores + denver2014scores + denver2015scores
footballdata.create_average_data(denverAverageStats,denverAverage)
denverAverageScores = footballdata.flatten_array(denverAverageScores)

detroitAverageStats = detroit2012stats + detroit2013stats + detroit2014stats + detroit2015stats
detroitAverageScores = detroit2012scores+ detroit2013scores + detroit2014scores + detroit2015scores
footballdata.create_average_data(detroitAverageStats,detroitAverage)
detroitAverageScores = footballdata.flatten_array(detroitAverageScores)

greenbayAverageStats = greenbay2012stats + greenbay2013stats + greenbay2014stats + greenbay2015stats
greenbayAverageScores = greenbay2012scores+ greenbay2013scores + greenbay2014scores + greenbay2015scores
footballdata.create_average_data(greenbayAverageStats,greenbayAverage)
greenbayAverageScores = footballdata.flatten_array(greenbayAverageScores)

houstonAverageStats = houston2012stats + houston2013stats + houston2014stats + houston2015stats
houstonAverageScores = houston2012scores+ houston2013scores + houston2014scores + houston2015scores
footballdata.create_average_data(houstonAverageStats,houstonAverage)
houstonAverageScores = footballdata.flatten_array(houstonAverageScores)

indianapolisAverageStats = indianapolis2012stats + indianapolis2013stats + indianapolis2014stats + indianapolis2015stats
indianapolisAverageScores = indianapolis2012scores+ indianapolis2013scores + indianapolis2014scores + indianapolis2015scores
footballdata.create_average_data(indianapolisAverageStats,indianapolisAverage)
indianapolisAverageScores = footballdata.flatten_array(indianapolisAverageScores)

jacksonvilleAverageStats = jacksonville2012stats + jacksonville2013stats + jacksonville2014stats + jacksonville2015stats
jacksonvilleAverageScores = jacksonville2012scores+ jacksonville2013scores + jacksonville2014scores + jacksonville2015scores
footballdata.create_average_data(jacksonvilleAverageStats,jacksonvilleAverage)
jacksonvilleAverageScores = footballdata.flatten_array(jacksonvilleAverageScores)

kansascityAverageStats = kansascity2012stats + kansascity2013stats + kansascity2014stats + kansascity2015stats
kansascityAverageScores = kansascity2012scores+ kansascity2013scores + kansascity2014scores + kansascity2015scores
footballdata.create_average_data(kansascityAverageStats,kansascityAverage)
kansascityAverageScores = footballdata.flatten_array(kansascityAverageScores)

losangelesAverageStats = losangeles2012stats + losangeles2013stats + losangeles2014stats + losangeles2015stats
losangelesAverageScores = losangeles2012scores+ losangeles2013scores + losangeles2014scores + losangeles2015scores
footballdata.create_average_data(losangelesAverageStats,losangelesAverage)
losangelesAverageScores = footballdata.flatten_array(losangelesAverageScores)

miamiAverageStats = miami2012stats + miami2013stats + miami2014stats + miami2015stats
miamiAverageScores = miami2012scores+ miami2013scores + miami2014scores +miami2015scores
footballdata.create_average_data(miamiAverageStats,miamiAverage)
miamiAverageScores = footballdata.flatten_array(miamiAverageScores)

minnesotaAverageStats = minnesota2012stats + minnesota2013stats + minnesota2014stats + minnesota2015stats
minnesotaAverageScores = minnesota2012scores+ minnesota2013scores + minnesota2014scores + minnesota2015scores
footballdata.create_average_data(minnesotaAverageStats,minnesotaAverage)
minnesotaAverageScores = footballdata.flatten_array(minnesotaAverageScores)

newenglandAverageStats = newengland2012stats + newengland2013stats + newengland2014stats + newengland2015stats
newenglandAverageScores = newengland2012scores+ newengland2013scores + newengland2014scores + newengland2015scores
footballdata.create_average_data(newenglandAverageStats,newenglandAverage)
newenglandAverageScores = footballdata.flatten_array(newenglandAverageScores)

neworleansAverageStats = neworleans2012stats + neworleans2013stats + neworleans2014stats + neworleans2015stats
neworleansAverageScores = neworleans2012scores+ neworleans2013scores + neworleans2014scores + neworleans2015scores
footballdata.create_average_data(neworleansAverageStats,neworleansAverage)
neworleansAverageScores = footballdata.flatten_array(neworleansAverageScores)

newyorkgAverageStats = newyorkg2012stats + newyorkg2013stats + newyorkg2014stats + newyorkg2015stats
newyorkgAverageScores = newyorkg2012scores+ newyorkg2013scores + newyorkg2014scores + newyorkg2015scores
footballdata.create_average_data(newyorkgAverageStats,newyorkgAverage)
newyorkgAverageScores = footballdata.flatten_array(newyorkgAverageScores)

newyorkjAverageStats = newyorkj2012stats + newyorkj2013stats + newyorkj2014stats + newyorkj2015stats
newyorkjAverageScores = newyorkj2012scores+ newyorkj2013scores + newyorkj2014scores + newyorkj2015scores
footballdata.create_average_data(newyorkjAverageStats,newyorkjAverage)
newyorkjAverageScores = footballdata.flatten_array(newyorkjAverageScores)

oaklandAverageStats = oakland2012stats + oakland2013stats + oakland2014stats + oakland2015stats
oaklandAverageScores = oakland2012scores+ oakland2013scores + oakland2014scores + oakland2015scores
footballdata.create_average_data(oaklandAverageStats,oaklandAverage)
oaklandAverageScores = footballdata.flatten_array(oaklandAverageScores)

philadelphiaAverageStats = philadelphia2012stats + philadelphia2013stats + philadelphia2014stats + philadelphia2015stats
philadelphiaAverageScores = philadelphia2012scores+ philadelphia2013scores + philadelphia2014scores + philadelphia2015scores
footballdata.create_average_data(philadelphiaAverageStats,philadelphiaAverage)
philadelphiaAverageScores = footballdata.flatten_array(philadelphiaAverageScores)

pittsburghAverageStats = pittsburgh2012stats + pittsburgh2013stats + pittsburgh2014stats + pittsburgh2015stats
pittsburghAverageScores = pittsburgh2012scores+ pittsburgh2013scores + pittsburgh2014scores + pittsburgh2015scores
footballdata.create_average_data(pittsburghAverageStats,pittsburghAverage)
pittsburghAverageScores = footballdata.flatten_array(pittsburghAverageScores)

sandiegoAverageStats = sandiego2012stats + sandiego2013stats + sandiego2014stats + sandiego2015stats
sandiegoAverageScores = sandiego2012scores+ sandiego2013scores + sandiego2014scores + sandiego2015scores
footballdata.create_average_data(sandiegoAverageStats,sandiegoAverage)
sandiegoAverageScores = footballdata.flatten_array(sandiegoAverageScores)

sanfranAverageStats = sanfran2012stats + sanfran2013stats + sanfran2014stats + sanfran2015stats
sanfranAverageScores = sanfran2012scores+ sanfran2013scores + sanfran2014scores + sanfran2015scores
footballdata.create_average_data(sanfranAverageStats,sanfranAverage)
sanfranAverageScores = footballdata.flatten_array(sanfranAverageScores)

seattleAverageStats = seattle2012stats + seattle2013stats + seattle2014stats + seattle2015stats
seattleAverageScores = seattle2012scores+ seattle2013scores + seattle2014scores + seattle2015scores
footballdata.create_average_data(seattleAverageStats,seattleAverage)
seattleAverageScores = footballdata.flatten_array(seattleAverageScores)

tampabayAverageStats = tampabay2012stats + tampabay2013stats + tampabay2014stats + tampabay2015stats
tampabayAverageScores = tampabay2012scores+ tampabay2013scores + tampabay2014scores + tampabay2015scores
footballdata.create_average_data(tampabayAverageStats,tampabayAverage)
tampabayAverageScores = footballdata.flatten_array(tampabayAverageScores)

tennesseeAverageStats = tennessee2012stats + tennessee2013stats + tennessee2014stats + tennessee2015stats
tennesseeAverageScores = tennessee2012scores+ tennessee2013scores + tennessee2014scores + tennessee2015scores
footballdata.create_average_data(tennesseeAverageStats,tennesseeAverage)
tennesseeAverageScores = footballdata.flatten_array(tennesseeAverageScores)

washingtonAverageStats = washington2012stats + washington2013stats + washington2014stats + washington2015stats
washingtonAverageScores = washington2012scores+ washington2013scores + washington2014scores + washington2015scores
footballdata.create_average_data(washingtonAverageStats,washingtonAverage)
washingtonAverageScores = footballdata.flatten_array(washingtonAverageScores)

##MAIN##

#print("average newengland stats")
#print(newenglandAverage)
#print("average NE scores")
#print(newenglandAverageScores)

#SVR
#svr = SVR(kernel='linear', C=1e3)
#svr.fit(newenglandAverage,newenglandAverageScores)
#print(newengland2016stats[0][0])
#a = np.reshape(newengland2016stats[0][0],(1,-1))
#print(newengland2016scores[0])
#print(svr.predict(a))

#train_average_offense(newenglandAverage,arizonaAverage)
print("new england 2016 stats")
print(newengland2016stats)

print("average newengland stats")
print(newenglandAverage)
print("average NE scores")
print(newenglandAverageScores)

print("create offense feature data")
newenglandFeatureset = create_offense_feature_data(newenglandAverageStats,newenglandAverage,teamsAverages)

print("new england train data")
newenglandtraindata =  newengland2014stats + newengland2015stats
print(newenglandtraindata)

svr_firstdowns = predict_offense_firstdowns(newenglandFeatureset,newenglandtraindata)
svr_passyards = predict_offense_passyards(newenglandFeatureset,newenglandtraindata)
svr_rushyards = predict_offense_rushyards(newenglandFeatureset,newenglandtraindata)
svr_turnovers = predict_offense_turnovers(newenglandFeatureset,newenglandtraindata)


#setup the 2016 game to predict
newenglandgame = newenglandAverage[63][0:4]
arizonagame = arizonaAverage[63][4:8]

game = [a - b for a, b in zip(newenglandgame,arizonagame)]
game = np.reshape(game,(1,-1))

predictedgame = []

print("predict first downs")
predictedgame.append(svr_firstdowns.predict(game))
print(predictedgame[0])
print("actual first downs")
print(newengland2016stats[0][0][4])

print("predict pass yards")
predictedgame.append(svr_passyards.predict(game))
print(predictedgame[1])
print("actual pass yards")
print(newengland2016stats[0][0][5])

print("predict rush yards")
predictedgame.append(svr_rushyards.predict(game))
print(predictedgame[2])
print("actual rush yards")
print(newengland2016stats[0][0][6])

print("predict turnovers")
predictedgame.append(svr_turnovers.predict(game))
print(predictedgame[3])
print("actual turnovers")
print(newengland2016stats[0][0][7])


#predict score with just offense stats
offenseaverage = []
flatnewenglandAverageStats = footballdata.flatten_array(newenglandAverageStats)
for game in flatnewenglandAverageStats:
    offenseaverage.append(game[4:8])

svrscore = SVR(kernel='linear', C=1e3)
svrscore.fit(offenseaverage,newenglandAverageScores)
predictedgame = np.reshape(predictedgame,(1,-1))
print("predict score")
print(svrscore.predict(predictedgame))
print("actual score")
print(newengland2016scores[0][0])