from sklearn import tree
from sklearn.svm import SVR
import numpy as np
import pandas as pd
import csv
import math
import os
import glob


#CSV files -> https://www.pro-football-reference.com/

#Panda Data gather
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

#totalOffense = pd.read_csv("teamOffense.csv")
#print(totalOffense)
#print(totalOffense.iloc[0])


gameHeaders = ["team score", "opp score", "O1stD", "OTotYd", "OPassY", "ORushY", "TO",
              "D1stD", "DTotYd", "DPassY", "DRushY", "DTO","Expected Offense Points",
              "Expected Defense Points", "Expected Special Points"]




#pull data out and seperate into scores and
#for index, row in pats2016.iterrows():
#    game = []

#    for x in range(12,17):
#        if math.isnan(row[x]):
#            game.append(0)
#        else:
#            game.append(row[x])

#    if game[0]==0 and game[1] == 0:
#        print('bye week game')
#    else:
#        newengland.append(game)
#        scores.append(row[10])

 


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

#1
#arizona = []
#2
#atlanta = []
#3
#baltimore = []
#4
#buffalo = []
#5
#carolina = []
#6
#chicago = []
#7
#cincinnati = []
#8
#cleveland = []
#9
#dallas = []
#10
#denver = []
#11
#detroit = []
#12
#greenbay = []
#13
#houston = []
#14
#indianapolis = []
#15
#jacksonville = []
#16
#kansascity = []
#17
#losangeles = []
#18
#miami = []
#19
#minnesota = []
#20
#newengland = []
#21
#neworleans = []
#22
#newyorkg = []
#23
#newyorkj = []
#24
#oakland = []
#25
#philadelphia = []
#26
#pittsburgh = []
#27
#sandiego = []
#28
#sanfran = []
#29
#seattle = []
#30
#tampabay = []
#31
#tennessee = []
#32
#washington = []

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
atlantas2015scores = []
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


scores2015 = [arizona2015scores ,atlantas2015scores,baltimore2015scores,buffalo2015scores,carolina2015scores,chicago2015scores,cincinnati2015scores,cleveland2015scores,dallas2015scores,denver2015scores,
              detroit2015scores,greenbay2015scores,houston2015scores,indianapolis2015scores,jacksonville2015scores,kansascity2015scores,losangeles2015scores,miami2015scores,minnesota2015scores,
              newengland2015scores,neworleans2015scores,newyorkg2015scores,newyorkj2015scores,oakland2015scores,philadelphia2015scores,pittsburgh2015scores,sandiego2015scores,sanfran2015scores,
              seattle2015scores,tampabay2015scores,tennessee2015scores,washington2015scores]

def getData(directory,teams,scores):
    i = 0
    for file in glob.iglob(directory):
        print(file)
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
                print('bye week game')
            else:
                season.append(game)
                seasonscores.append(row[10])
        teams[i].append(season)
        scores[i].append(seasonscores)
        print("team 2015")
        print(teams[i])
        print("scores 2015")
        print(scores[i])
        print(i)
        i += 1

getData(directory2015,teams2015, scores2015)