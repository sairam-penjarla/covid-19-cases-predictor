import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
data = pd.read_csv('owid-india-covid-data.csv', sep=',')
data = data[['id', 'total_cases']]
data.fillna(data.ffill(axis=0), inplace=True)
x = np.array(data['id']).reshape(-1, 1)
y = np.array(data['total_cases']).reshape(-1, 1)
polyfeature = PolynomialFeatures(degree=6)
x = polyfeature.fit_transform(x)
model = linear_model.LinearRegression()
model.fit(x, y)
accuracy = model.score(x, y)
print(f'Accuracy:{round(accuracy * 100, 3)}%')
y0 = model.predict((x))
days = 2
plt.plot(y0, 'red')
plt.plot(y, 'purple')
plt.legend(["Actual", "Predicted"], loc ="lower right")
plt.show()
print(f'Prediction - Cases after {days} days: ',end='')
prediction = float(y0[-2]/1000000)
print(round(prediction, 2),'M')
