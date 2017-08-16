from sklearn import tree
from sklearn.svm import SVR
import numpy as np
import pandas as pd
import csv
import math
#fruit prediction
features = [[140,1], [130,1], [150,1], [170,0]]
labels = [0,0,1,1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,labels)
#print (clf.predict([[135,0]]))


#totalOffenseHeaders = ["rank 2016", "team", "games", "points scored", "total yards", "offensive plays", "total yards per play", "turnovers lost", "fumbles lost", "total first downs", 
#                       "passing completions", "passing attempts","passing yards","passing touchdowns","interceptions", "net yards per pass", "first downs passing", 
#                       "attempted rushing plays", "yards rushing", "rushing touchdowns", "rushing yards per attempt", "rushing first downs", 
#                       "penalties","penalty yards", "first downs by penalty", 
#                       "scoring drive percentage", "turnover percentage", "expected points contributed"]

#Numpy Data gather
#totalOffense = np.genfromtxt('teamOffense.csv',
#                             dtype="i8,S5,i8,i8,i8,i8,f8,i8,i8,i8,i8,i8,i8,i8,i8,f8,i8,i8,i8,i8,f8,i8,i8,i8,i8,f8,f8,f8",
#                             names= totalOffenseHeaders,
#                             delimiter=',')
#print(totalOffense)


#CSV files -> https://www.pro-football-reference.com/

#Panda Data gather
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

#totalOffense = pd.read_csv("teamOffense.csv")
#print(totalOffense)
#print(totalOffense.iloc[0])


pats2016 = pd.read_csv("NE2016.csv")

gameHeaders = ["team score", "opp score", "O1stD", "OTotYd", "OPassY", "ORushY", "TO", 
              "D1stD", "DTotYd", "DPassY", "DRushY", "DTO","Expected Offense Points", 
              "Expected Defense Points", "Expected Special Points"]

#19
newengland = []
scores = []

#pull data out and seperate into scores and 
for index, row in pats2016.iterrows():
    game = []
    for x in range(10,22):
        if x==10:
            if math.isnan(row[x]):
                print('bye week scores')
            else:
                scores.append(row[x])
        else:
            if math.isnan(row[x]):
                game.append(0)
            else:
                game.append(row[x])
    if game[0]==0 and game[1] == 0:
        print('bye week game')
    else:
        newengland.append(game)

for row in scores:
    print(row)

for row in newengland:
    print(row)


#Create Support Vector Regressions
#SVR is to predict next value in a series

svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3)
svr_rbf = SVR(kernel='rbf', C=1e3, gamma = 0.1)

#should we use Support Vector Machines as we are trying to predict scores?
#setup Support Vector Regression
svr_lin.fit(newengland,scores)
svr_poly.fit(newengland,scores)
svr_rbf.fit(newengland,scores)


#predict 2015 Redskins game 27-10
print('2015 vs Redskins 27-10 win')

print('Lin Fit')
print(svr_lin.predict([10,27,460,299,161,2,16,250,213,37,2]))

print('Poly Fit')
print(svr_poly.predict([10,27,460,299,161,2,16,250,213,37,2]))

print('RBF Fit')
print(svr_rbf.predict([10,27,460,299,161,2,16,250,213,37,2]))

#predict 2015 Broncos game 24-30 loss
print('2015 Broncos game 24-30 loss')

print('Lin Fit')
print(svr_lin.predict([30,16,301,262,39,1,23,433,254,179,1]))

print('Poly Fit')
print(svr_poly.predict([30,16,301,262,39,1,23,433,254,179,1]))

print('RBF Fit')
print(svr_rbf.predict([30,16,301,262,39,1,23,433,254,179,1]))

#team arrays
#should we define teams as numbers in code?
#1
arizona = []
#2
atlanta = []
#3
baltimore = []
#4
buffalo = []
#5
carolina = []
#6
chicago = []
#7
cincinnati = []
#8
cleveland = []
#9
dallas = []
#10
denver = []
#11
detroit = []
#12
greenbay = []
#13
houston = []
#14
indianapolis = []
#15
jacksonville = []
#16
kansascity = []
#17
miami = []
#18
minnesota = []

#20
neworleans = []
#21
newyorkg = []
#22
newyorkj = []
#23
oakland = []
#24
philadelphia = []
#25
pittsburgh = []
#26
sandiego = []
#27
sanfrancisco = []
#28
seattle = []
#29
stlouis = []
#30
tampabay = []
#31
tennessee = []
#32
washington = []

teams = [arizona,atlanta,baltimore,buffalo,carolina,chicago,cincinnati,cleveland,dallas,denver,detroit,greenbay,houston,indianapolis,
         jacksonville,kansascity,miami,minnesota,newengland,neworleans,newyorkg,newyorkj,oakland,philadelphia,pittsburgh,sandiego,sanfrancisco,
         seattle,stlouis,tampabay,tennessee,washington]