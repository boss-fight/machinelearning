from sklearn import tree
import numpy as np
import pandas as pd
import csv

#decision making
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

totalOffense = pd.read_csv("teamOffense.csv")
#print(totalOffense)
print(totalOffense.iloc[0])

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
#19
newengland = []
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