import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('top5_leagues_player.csv')
df = df. dropna()
df = df.drop(["full_name", "height", "place_of_birth", "shirt_nr", "foot", "joined_club", "player_agent", "outfitter"], axis = 1)
df = df.sort_values(by = ["price"], ascending=False)

rslt_df = df[df['price'] < 60]
rslt_df = rslt_df[rslt_df['price'] > 20]
rslt_df = rslt_df[rslt_df['age'] < 25]
rslt_df = rslt_df[rslt_df['price'] < rslt_df['max_price']]
rslt_df.dropna()
print(rslt_df)

'''
# add 2 columns using + operator
df['Goals Per Game'] = df['Home Team Goals'] + df['Away Team Goals']
#df['Goals For']=df


df = df.drop(['City','Stage','Half-time Home Goals','Half-time Away Goals', 'Stadium', 'Datetime', 'Win conditions','Attendance','Referee','Assistant 1','Assistant 2','RoundID','MatchID','Home Team Initials','Away Team Initials'], axis=1)
df.head(15)
'''