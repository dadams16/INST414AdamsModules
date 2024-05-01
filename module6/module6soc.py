import json
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



data = pd.read_csv("salary_goals.csv")
data = data[data['current_value'] < 100000000]
data = data[data['current_value'] > 10000000]
data['current_value']= data['current_value'] / 1000000

attackers = data[data['position'].str.contains("Centre-Forward")]
attackers = attackers.reset_index(drop = True)

labels = attackers['current_value']

goals = (attackers['goals'] * attackers['appearance']) #Works as expected
assists = (attackers['assists'] * attackers['appearance']) #Works as expected
g_a = (goals + assists)
features = g_a

attackers['goals'] = goals
attackers['assists'] = assists

features = pd.DataFrame(list(zip(goals, assists)), columns = ['goals', 'assists'])

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
prediction = model.predict(X_test)
pred_vs_true = pd.DataFrame(list(zip(attackers['name'], labels, prediction, goals, assists, attackers['age'], attackers['appearance'], attackers['minutes played'])), columns=['Name', 'True Value', 'Predicted Value', 'goals', 'assists', 'age', 'appearances', 'minutes_played'])

result_df = pred_vs_true[pred_vs_true['True Value'] < pred_vs_true['Predicted Value']]
print(result_df.head(20))
