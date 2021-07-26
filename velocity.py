import pandas as pd
import numpy as np
import csv
import plotly_express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("velocity.csv")

velocityList = df["Velocity"].to_list()
escapeList = df["Escaped"].to_list()

velocityArray = np.array(velocityList)
escapeArray = np.array(escapeList)

b, a = np.polyfit(velocityArray, escapeArray, 1)
y = []



for x in velocityArray:
    y_value = b*x + a
    y.append(y_value)

# fig = px.scatter(df, x = velocityList, y = escapeList, title = "Temperature Rates")
# fig.update_layout(shapes = [
#     dict(
#         type = 'line',
#         y0 = min(escapeArray),
#         y1 = max(escapeArray),
#         x0 = min(velocityArray),
#         x1 = max(velocityArray)
#     )
# ])
# fig.show()

X = np.reshape(velocityList, (len(velocityList), 1))
Y = np.reshape(escapeList, (len(escapeList), 1))

lr = LogisticRegression()
lr.fit(X, Y)

plt.figure()
plt.scatter(X.ravel(), Y, color='black', zorder=20)

def model(x):
  return 1 / (1 + np.exp(-x))

# Using the line formula 
X_test = np.linspace(0, 100, 200)
chances = model(X_test * lr.coef_ + lr.intercept_).ravel()

# plt.plot(X_test, chances, color='red', linewidth=3)
# plt.axhline(y=0, color='k', linestyle='-')
# plt.axhline(y=1, color='k', linestyle='-')
# plt.axhline(y=0.5, color='b', linestyle='--')

# # do hit and trial by changing the value of X_test
# plt.axvline(x=X_test[65], color='b', linestyle='--')

# plt.ylabel('y')
# plt.xlabel('X')
# plt.xlim(75, 85)
# plt.show()

velocity_input = float(input("Enter Your Velocity Here: "))
chances = model(velocity_input * lr.coef_ + lr.intercept_).ravel()[0]
if chances <= 0.01:
  print("The person will not escape.")
elif chances >= 1:
  print("The person will escape!")
elif chances < 0.5:
  print("The person may not escape.")
else:
  print("The person may escape.")