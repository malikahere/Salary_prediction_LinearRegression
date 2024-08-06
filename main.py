import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Importing the dataset
db = pd.read_csv('Salary.csv')
x = db.iloc[: ,:-1].values
y = db.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_tr , x_tst , y_tr , y_tst = train_test_split(x , y , test_size=0.2 , random_state=1)


## Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_tr , y_tr)


## Predicting the Test set results
y_pred = reg.predict(x_tst)
y_reg = reg.predict(x_tr)


## Visualising the Training set results
plt.scatter(x_tr , y_tr , color = 'pink')
plt.plot(x_tr , y_reg, color ='blue')
plt.title("Salary_as_per_experience(Training)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()


## Visualising the Test set results
plt.scatter(x_tst , y_tst , color = 'pink')
plt.plot(x_tr , y_reg, color ='blue')
plt.title("Salary_as_per_experience(Test)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()


## Let's predict a salary of an employee with number of years of experience
years = float(input("experience of years:"))
pred =reg.predict([[years]])
print ("Salary with", years ,"years of experience is", pred )
              