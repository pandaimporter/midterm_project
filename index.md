### 310 Midterm:  

## Topics: Midterm questions on random forests and multivariate regressors.  

# Project Details:  
> This project's idea was to take a weather history dataset and analyze it to find relationships between one or more features of the data and a target feature. Common relationships
were between temperature, apparent temperature, humidity, pressure, wind bearing, and wind speed.  
> Regression methods presented in class were used to find these relationships. Linear, Ridge, Polynomial, and Random Forest Regressions were used, in particular.  

# Notable Code Segments, Answers, and Justifications:  

This project involved a lot of k-fold validation. So, I modified a k-fold function to compute only what I needed to speed up the process and eliminate unnecessary coding,
and had a very large import statement at the beginning to avoid ambiguity in the code.

```markdown
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
SS = StandardScaler()
```

This K-Fold function was used to speed up the coding process. It returns the average RMSE after a k-fold validation process.  

```markdown
def DoKFold(X,y,model,k, rs, st=False):
  SE = []
  kf = KFold(n_splits=k,shuffle=True,random_state=rs)
  for idxtrain,idxtest in kf.split(X):
    Xtrain = X[idxtrain,:]
    Xtest  = X[idxtest,:]
    ytrain = y[idxtrain]
    ytest  = y[idxtest]
    if(st == True):
      Xtrain = SS.fit_transform(Xtrain)
    model.fit(Xtrain,ytrain)
    yhat = model.predict(Xtest)
    SE.append(np.sqrt(MSE(ytest,yhat)))
  return np.mean(SE)
```

The first notable segment was problem 5, which addressed the RMSE for random forest regression for 100 trees, depth 50, and a random state of 1693.  
The main feature here was apparent temperature, which required reshaping, and the target here was humidity. This data was then passed along with a random forest
regressor, rfr, to a 10-fold validation function.

```markdown
rfr = RandomForestRegressor(n_estimators=100, max_depth=50)
x = data['Apparent Temperature (C)'].values.reshape(-1, 1)
y = data['Humidity']
k = 10
DoKFold(x, y, rfr, k, rs=1693)
```
In my code, this produced a RMSE of .143524.  

The next segment was problem 8, which addressed the RMSE for a multivariate polynomial regreesion. Degree was 6 and the random state was 1234.   
The feature matrix x included four specified features from the data, and the target was temperature. The PolynomialFeatures library was used to transform the data into
the desired format, and was put into a linear regression model, pfr2. All of these were then passed to a 10-fold validation function.

```markdown
polynomial_features= PolynomialFeatures(degree=6)
X = data[['Humidity','Wind Speed (km/h)','Pressure (millibars)','Wind Bearing (degrees)']] 
Xp = polynomial_features.fit_transform(X)
pfr2 = LinearRegression()

y = data['Temperature (C)']
k = 10
rs = 1234

DoKFold(Xp, y, pfr2, k, rs)
```
In my code, this produced a RMSE of 6.1270.  
  
The last segment was problem 9, which addressed the RMSE for a multivariate random forest model. The conditions are the same as problem 5, but instead with a random
state of 1234.

```markdown
rfr2 = RandomForestRegressor(n_estimators=100, max_depth=50)
X = data[['Humidity','Wind Speed (km/h)','Pressure (millibars)','Wind Bearing (degrees)']].values
y = data['Temperature (C)']

k = 10
rs = 1234

DoKFold(X, y, rfr2, k, rs)
```
In my code, this produced a RMSE of 5.8331.  
