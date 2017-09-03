from sklearn import tree
from sklearn.svm import SVR
from sklearn import preprocessing
import numpy as np
import pandas as pd
import csv
import math
import glob


#CSV files -> https://www.pro-football-reference.com/

#Panda Data gather
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

gameHeaders = ["team score", "opp score", "O1stD", "OTotYd", "OPassY", "ORushY", "TO",
              "D1stD", "DTotYd", "DPassY", "DRushY", "DTO","Expected Offense Points",
              "Expected Defense Points", "Expected Special Points"]

#Create Support Vector Regressions
#SVR is to predict next value in a series

#svr_lin = SVR(kernel='linear', C=1e3)
#print ('svr-lin')
#svr_poly = SVR(kernel='poly', C=1e3)
#print ('svr_poly')
#svr_rbf = SVR(kernel='rbf', C=1e3, gamma = 0.1)
#print ('svr_rbf')

#should we use Support Vector Machines as we are trying to predict scores?
#setup Support Vector Regression

#svr_lin.fit(newengland,scores)
#print ('fit svr_lin')
#svr_poly.fit(newengland,scores)
#print ('fit svr_poly')
#svr_rbf.fit(newengland,scores)
#print ('fit svr_rbf')


#predict 2015 Redskins game 27-10
#print('2015 vs Redskins 27-10 win')

#print('Lin Fit')
#x = [27,460,299,161,2]
#x = np.reshape(x,(1, -1))
#print(svr_lin.predict(x))

#print('Poly Fit')
#print(svr_poly.predict(x))

#print('RBF Fit')
#print(svr_rbf.predict(x))

#predict 2015 Broncos game 24-30 loss
#print('2015 Broncos game 24-30 loss')

#print('Lin Fit')
#x = [16,101,262,39,1]
#x = np.reshape(x,(1, -1))
#print(svr_lin.predict(x))

#print('Poly Fit')
#print(svr_poly.predict(x))

#print('RBF Fit')
#print(svr_rbf.predict(x))

#team arrays

#0
#arizona = []
#1
#atlanta = []
#2
#baltimore = []
#3
#buffalo = []
#4
#carolina = []
#5
#chicago = []
#6
#cincinnati = []
#7
#cleveland = []
#8
#dallas = []
#9
#denver = []
#10
#detroit = []
#11
#greenbay = []
#12
#houston = []
#13
#indianapolis = []
#14
#jacksonville = []
#15
#kansascity = []
#16
#losangeles = []
#17
#miami = []
#18
#minnesota = []
#19
#newengland = []
#20
#neworleans = []
#21
#newyorkg = []
#22
#newyorkj = []
#23
#oakland = []
#24
#philadelphia = []
#25
#pittsburgh = []
#26
#sandiego = []
#27
#sanfran = []
#28
#seattle = []
#29
#tampabay = []
#30
#tennessee = []
#31
#washington = []

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

#pull data from stats folder method
def getData(directory,teams,scores):
    i = 0
    for file in glob.iglob(directory):
        seasonfile = pd.read_csv(file)
        season = []
        seasonscores = []
        for index, row in seasonfile.iterrows():
            game = []
            for x in range(12,17):
                if math.isnan(row[x]):
                    game.append(0)
                else:
                    game.append(row[x]) 
        
            if game[0]==0 and game[1] == 0:
                b = 0
            else:
                season.append(game)
                seasonscores.append(row[10])
        teams[i].append(season)
        scores[i].append(seasonscores)
        i += 1

#train N games
def trainData(teamStats, teamScores, numberOfGames, svr):
    #create range of games you want based off number of games
    range1 = len(teamStats[0])+len(teamStats[1])-numberOfGames
    range2 = len(teamStats[0])+len(teamStats[1])
    newTeamStats = []
    for season in teamStats:
        for game in season:
            newTeamStats.append(game)
    
    newerTeamStats = []
    for game in newTeamStats[range1:range2]:
        newerTeamStats.append(game)

    newTeamScores = []
    for season in teamScores:
        for game in season:
            newTeamScores.append(game)
    
    newerTeamScores = []
    for game in newTeamScores[range1:range2]:
        newerTeamScores.append(game)

    svr.fit(newerTeamStats,newerTeamScores)

def season_walkthrough(teamStats,teamScores, teamStats2016, teamScores2016, svr):
    #original trainData from above method minus range parameter
    newTeamStats = []
    for season in teamStats:
        for game in season:
            newTeamStats.append(game)
    
    newerTeamStats = []
    for game in newTeamStats:
        newerTeamStats.append(game)

    newTeamScores = []
    for season in teamScores:
        for game in season:
            newTeamScores.append(game)
    
    newerTeamScores = []
    for game in newTeamScores:
        newerTeamScores.append(game)

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

        #refit model with new data
        svr.fit(newerTeamStats, newerTeamScores)



#create team arrays    
getData(directory2014,teams2014, scores2014)
getData(directory2015,teams2015, scores2015)
getData(directory2016,teams2016, scores2016)

#combine seasons
newenglandTrainStats = newengland2014stats + newengland2015stats
newenglandTrainScores = newengland2014scores + newengland2015scores

#SVR
svr_lin = SVR(kernel='linear', C=1e3)

season_walkthrough(newenglandTrainStats,newenglandTrainScores,newengland2016stats,newengland2016scores,svr_lin)