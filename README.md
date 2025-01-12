(Return to Home)[https://04mscott.github.io]
# Housing-Price-Prediction-Model-

In this project I used multiple forms of regression from the SciKit-Learn library within a Jupyter environment

Importing Data + Data Cleaning
```Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from google.colab import drive
drive.mount('/content/drive')
     
Mounted at /content/drive

df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Personal Project/data/kc_house_data.csv')
     
df.columns
```     
Index(['id', 'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long', 'sqft_living15', 'sqft_lot15'],
      dtype='object')
```Python
df['date'] = pd.to_datetime(df['date'])
df.head()
```
id	date	price	bedrooms	bathrooms	sqft_living	sqft_lot	floors	waterfront	view	...	grade	sqft_above	sqft_basement	yr_built	yr_renovated	zipcode	lat	long	sqft_living15	sqft_lot15
0	7129300520	2014-10-13	221900.0	3	1.00	1180	5650	1.0	0	0	...	7	1180	0	1955	0	98178	47.5112	-122.257	1340	5650
1	6414100192	2014-12-09	538000.0	3	2.25	2570	7242	2.0	0	0	...	7	2170	400	1951	1991	98125	47.7210	-122.319	1690	7639
2	5631500400	2015-02-25	180000.0	2	1.00	770	10000	1.0	0	0	...	6	770	0	1933	0	98028	47.7379	-122.233	2720	8062
3	2487200875	2014-12-09	604000.0	4	3.00	1960	5000	1.0	0	0	...	7	1050	910	1965	0	98136	47.5208	-122.393	1360	5000
4	1954400510	2015-02-18	510000.0	3	2.00	1680	8080	1.0	0	0	...	8	1680	0	1987	0	98074	47.6168	-122.045	1800	7503
5 rows × 21 columns

```Python
df.info()
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 21613 entries, 0 to 21612
Data columns (total 21 columns):
 #   Column         Non-Null Count  Dtype         
---  ------         --------------  -----         
 0   id             21613 non-null  int64         
 1   date           21613 non-null  datetime64[ns]
 2   price          21613 non-null  float64       
 3   bedrooms       21613 non-null  int64         
 4   bathrooms      21613 non-null  float64       
 5   sqft_living    21613 non-null  int64         
 6   sqft_lot       21613 non-null  int64         
 7   floors         21613 non-null  float64       
 8   waterfront     21613 non-null  int64         
 9   view           21613 non-null  int64         
 10  condition      21613 non-null  int64         
 11  grade          21613 non-null  int64         
 12  sqft_above     21613 non-null  int64         
 13  sqft_basement  21613 non-null  int64         
 14  yr_built       21613 non-null  int64         
 15  yr_renovated   21613 non-null  int64         
 16  zipcode        21613 non-null  int64         
 17  lat            21613 non-null  float64       
 18  long           21613 non-null  float64       
 19  sqft_living15  21613 non-null  int64         
 20  sqft_lot15     21613 non-null  int64         
dtypes: datetime64[ns](1), float64(5), int64(15)
memory usage: 3.5 MB
```Python
df.isna().sum()
```  
0
id	0
date	0
price	0
bedrooms	0
bathrooms	0
sqft_living	0
sqft_lot	0
floors	0
waterfront	0
view	0
condition	0
grade	0
sqft_above	0
sqft_basement	0
yr_built	0
yr_renovated	0
zipcode	0
lat	0
long	0
sqft_living15	0
sqft_lot15	0

dtype: int64
```Python
df = df.drop(columns=['id'])
     

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday
```
***Linear Regression***
```Python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
     

X = df.drop(columns=['date', 'price'])
y = df['price']
     

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)
     

scaler = StandardScaler()
     

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
     

lr = LinearRegression()
     

lr.fit(X_train, y_train)
```     
LinearRegression()
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
```Python
y_pred = lr.predict(X_test)
     

print(mean_absolute_error(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))
```     
123224.92269170494
33957469163.087692
0.7070935240287654

**Random Forest Regression**
```Python
rfr = RandomForestRegressor()
     

rfr.fit(X_train, y_train)
```     
RandomForestRegressor()
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
```Python
y_pred_rfr = rfr.predict(X_test)
     

print(mean_absolute_error(y_test, y_pred_rfr))
print(mean_squared_error(y_test, y_pred_rfr))
print(r2_score(y_test, y_pred_rfr))
```     
68147.32523664123
13253078421.293613
0.8856831032516632
**Gradient Booster Regression**
```Python
gbr = GradientBoostingRegressor()
     

gbr.fit(X_train, y_train)
```     
GradientBoostingRegressor()
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
```Python
y_pred_gbr = gbr.predict(X_test)
     

print(mean_absolute_error(y_test, y_pred_gbr))
print(mean_squared_error(y_test, y_pred_gbr))
print(r2_score(y_test, y_pred_gbr))
```     
77082.5534469952
15982086856.46248
0.8621435326257703
Stacking Regressor
```Python
from sklearn.ensemble import StackingRegressor, VotingRegressor
     

estimators = [
    ('rfr', rfr),
    ('gbr', gbr),
    ('ridge', ridge),
]
     

sr = StackingRegressor(
    estimators = estimators,
    final_estimator = lr,
)
     

sr.fit(X_train, y_train)
     
StackingRegressor(estimators=[('rfr', RandomForestRegressor()),
                              ('gbr', GradientBoostingRegressor()),
                              ('ridge', Ridge())],
                  final_estimator=LinearRegression())
```
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
```Python
y_pred_sr = sr.predict(X_test)
     

print(mean_absolute_error(y_test, y_pred_sr))
print(mean_squared_error(y_test, y_pred_sr))
print(r2_score(y_test, y_pred_sr))
```     
69635.53805445331
13766280176.490238
0.8812563859113646
```Python
vr = VotingRegressor([
    ('rfr', rfr),
    ('gbr', gbr),
    ('ridge', ridge),
])
     

vr.fit(X_train, y_train)
     
VotingRegressor(estimators=[('rfr', RandomForestRegressor()),
                            ('gbr', GradientBoostingRegressor()),
                            ('ridge', Ridge())])
```
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
```Python
y_pred_vr = vr.predict(X_test)
     

print(mean_absolute_error(y_test, y_pred_sr))
print(mean_squared_error(y_test, y_pred_sr))
print(r2_score(y_test, y_pred_sr))
```
     
69635.53805445331
13766280176.490238
0.8812563859113646

(Return to Home)[https://04mscott.github.io]
