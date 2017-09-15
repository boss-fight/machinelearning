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
                if x == 13:
                    continue
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


def create_average_data(teamStats,teamAverageStats):
    tempAverage = []
    for season in teamStats:
        for game in season:
            tempAverage.append(game)
            a = np.array(tempAverage)
            a = a.mean(axis=0)
            x = list(a)
            #rounding not sure if we need to
            x = [round(float(i),2) for i in x]
            teamAverageStats.append(x)




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
        svr.fit(newerTeamStats,newerTeamScores)



##MAIN##

#create team arrays    

getData(directory2012,teams2012, scores2012)
getData(directory2013,teams2013, scores2013)
getData(directory2014,teams2014, scores2014)
getData(directory2015,teams2015, scores2015)
getData(directory2016,teams2016, scores2016)



#combine seasons
newenglandTrainStats = newengland2012stats + newengland2013stats
newenglandTrainScores = newengland2012scores + newengland2013scores

create_average_data(newenglandTrainStats,newenglandAverage)

#SVR
#svr_lin = SVR(kernel='linear', C=1e3)
#season_walkthrough(newenglandTrainStats,newenglandTrainScores,newengland2016stats,newengland2016scores,svr_lin)
