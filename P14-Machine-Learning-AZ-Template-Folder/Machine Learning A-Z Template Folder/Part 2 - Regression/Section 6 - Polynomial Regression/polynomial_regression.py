 # Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
# On a small Dataset, it's not worth testing

# Feature Scaling, Current Library accounts for Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Fitting Linear Regression to the Dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Fitting Polynomial Regression to the Dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#Visualizing Linear Regression Results
#Real Observations
plt.scatter(X, y, color = 'red')
#Predictions
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth or Bluff')
plt.xlabel('Position Level')
plt.ylabel('Salary')

#Visualizing Polynomial Regression Results
#Complex Chart
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
#Real Observations
plt.scatter(X, y, color = 'red')
#Predictions
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff (Poly)')
plt.xlabel('Position Level')
plt.ylabel('Salary')

#Liner Prediction
y_pred = lin_reg.predict(np.array([6.5]).reshape(1, -1))

#Polynomial Prediction
y_pred_poly = lin_reg_2.predict(poly_reg.fit_transform(np.array([6.5]).reshape(1, -1)))




















