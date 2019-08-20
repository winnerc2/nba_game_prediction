##  Caleb Winner
## CSE 482 Class Project
## Objective: Predict NBA Basketball game outcomes
## Note: Code in runDecTree and runLogReg are heavily influenced by
##      the released code in exercise 6 from Tyler Derr, the CSE 482 TA for F18

import pandas as p
import sklearn as sk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import vars # file holding my constant vars

TrainMatchFile = '../matchups/matchups16-17.csv'
TestMatchFile = '../matchups/matchups17-18.csv'
# writes a new matchup csv file adding column for label 'winner'
def labelWinners(filename):
    MatchData = p.read_csv(filename)

    for i,row in MatchData.iterrows():
        vpts = row['PTS']
        hpts = row['PTS.1']
        vwin = 0
        hwin = 0
        if (vpts > hpts):
            vwin = 1
        else:
            hwin = 1
        MatchData.at[i,'visitorWin'] = vwin
        MatchData.at[i,'homeWin'] = hwin
    filename = filename[:-4] + '-labeled.csv'
    MatchData.to_csv(filename)
    return

def build_against_dict(filenameList,year):
    data = dict()
    for fname in filenameList:
        key = fname[5:-4]
        fname = '../data' + year + '/' + fname
        data[key] = p.read_csv(fname)
    return data

def build_year_long_dict(filenameList):
    data = dict()
    data['16-17'] = p.read_csv(filenameList[0])
    data['17-18'] = p.read_csv(filenameList[1])
    return data

def build_match_data_dict():
    data = dict()
    data['train'] = p.read_csv('../matchups/matchups16-17-labeled.csv')
    data['test'] = p.read_csv('../matchups/matchups17-18-labeled.csv')
    return data

# adds columns and values of the stats I will try to classify with.
# adds visitorSeasonW/l, homeSeasonW/l, visitorw/L as visitor, homeW/l as home,
#       visitorW/l within current matchup, homeW/l within current matchup
# inputs: labeled matchup dataframe, season stat dict, teams vs specific teams dict
def add_classifiers(raw_match_data, year_long_data, teams_against_data):
    final_data = raw_match_data
    final_data['winner'] = 0
    for i,row in raw_match_data.iterrows():
        vteam = row['Visitor']
        hteam = row['Home']
        vTotWL = get_stat(year_long_data,'TEAM', vteam, 'WIN%')
        hTotWL = get_stat(year_long_data,'TEAM', hteam, 'WIN%')
        visvisWL = calcVisitorWL(vteam)
        homehomeWL = calcHomeWL(hteam)
        vMatchupWL = get_stat(teams_against_data[vars.nameMapping[hteam]], 'TEAM', vteam, 'WIN%')
        hMatchupWL = get_stat(teams_against_data[vars.nameMapping[vteam]], 'TEAM', hteam, 'WIN%')
        # print("visitors: " + vteam)
        # print(vMatchupWL)
        # print("home: " + hteam)
        # print(hMatchupWL)
        try:
            final_data.at[i,'visTotWL'] = vTotWL
            final_data.at[i,'homeTotWL'] = hTotWL
            final_data.at[i,'visvisWL'] = visvisWL
            final_data.at[i,'homehomeWL'] = homehomeWL
            final_data.at[i,'visMatchupWL'] = vMatchupWL
            final_data.at[i,'homeMatchupWL'] = hMatchupWL
        except ValueError:
            print('i value: ' + str(i))
            print("visiting Team: " + vteam)
            print("home Team: " + hteam)
            print("hMatchupWL: " + str(hMatchupWL))
            print()
        if (row['visitorWin']):
            final_data.at[i, 'winner'] = 0  # 0 means visitors win
        else:
            final_data.at[i, 'winner'] = 1  # 1 means visitors win

    return final_data

# removes labels which the classifier would key on and instantly know the winner
def remove_extra_labels(df):
    df = df.drop(['Visitor', 'Home', 'PTS', 'PTS.1', 'visitorWin', 'homeWin'], axis=1)
    return df

# accesser to more easily get a value from my dicts
#inputs: frame: dataframe holding stats
#        compCol: col which to argue on
#       compVal: value of col used to select specific row
#       tarVal: column of value to get from row selected by prev inputs
def get_stat(frame, compCol, compVal, tarVal):
    return frame.loc[frame[compCol] == compVal][tarVal]

# calculates the visiting teams w/l as a visitor
def calcVisitorWL(visitingTeam):
    visitorOutcomes = get_stat(matchup_data['train'], 'Visitor', visitingTeam, 'visitorWin')
    visitorWinsCount = visitorOutcomes.sum()
    #print("visiting wins: " + str(visitorWinsCount))
    #losses = visitorOutcomes.size - visitorWinsCount
    return (visitorWinsCount / visitorOutcomes.size)

# calculates the home teams w/l at home
def calcHomeWL(homeTeam):
    homeOutcomes = get_stat(matchup_data['train'], 'Home', homeTeam, 'homeWin')
    homeWinsCount = homeOutcomes.sum()
    #print('home wins: ' + str(homeWinsCount))
    return (homeWinsCount / homeOutcomes.size)

def runDecTree(dfTrain, dfTest):
    # maxdepths = [1, 5, 10, 50, 100]
    # I found best to be either 1 or 2... Selecting 2
    Y = dfTrain['winner']
    X = dfTrain.drop('winner', axis=1)
    Y2 = dfTest['winner']
    X2 = dfTest.drop('winner', axis=1)
    clf = tree.DecisionTreeClassifier(max_depth=2)
    clf = clf.fit(X, Y)
    Ypred = clf.predict(X2)
    treeacc = accuracy_score(Y2, Ypred)
    print(treeacc)
    return

def runLogReg(dfTrain, dfTest):
    # regularizers = [0.001, 0.01, 0.1, 0.5, 1]
    # I found best to be .5 or 1... Selecting 1
    Y = dfTrain['winner']
    X = dfTrain.drop('winner', axis=1)
    Y2 = dfTest['winner']
    X2 = dfTest.drop('winner', axis=1)
    clf = LogisticRegression(C=1, solver='liblinear')
    clf = clf.fit(X, Y)
    Ypred = clf.predict(X2)
    log_acc = accuracy_score(Y2, Ypred)
    print(log_acc)
    return

def main():
    # build dicts of data
    teams_against_data = dict() # This dict holds matchup data for each team.  Ex: all teams vs Detroit Pistons
    teams_against_data['16-17'] = build_against_dict(vars.teamNames,'16-17')
    teams_against_data['17-18'] = build_against_dict(vars.teamNames,'17-18')
    year_long_data = build_year_long_dict(vars.yearFiles) # This dict holds overall data for a whole season
    final_data_train = add_classifiers(matchup_data['train'], year_long_data['16-17'], teams_against_data['16-17'])
    print("starting test data build")
    final_data_test = add_classifiers(matchup_data['test'], year_long_data['17-18'], teams_against_data['17-18'])

    final_data_train = remove_extra_labels(final_data_train)
    final_data_test = remove_extra_labels(final_data_test)
    #final_data_train.to_csv('finaldata-test.csv')
    final_data_both = p.concat([final_data_train, final_data_test])
    print("running Decision Tree...")
    runDecTree(final_data_train, final_data_test)
    print("\n\nrunning LogisticRegression...\n")
    runLogReg(final_data_train, final_data_test)



# below 2 commands read the matchup files and label the winners
#labelWinners(TrainMatchFile)
#labelWinners(TestMatchFile)
matchup_data = build_match_data_dict() # Main dict which holds every game matchup in from 16-17 and 17-18 season
main()
