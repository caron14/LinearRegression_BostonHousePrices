import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
dataset = load_boston()
sns.set()


df = pd.DataFrame(dataset.data)
df.columns = dataset.feature_names
df["PRICES"] = dataset.target

# Show the relation, "PRICES" vs each variable
df.plot(x="TAX", y="PRICES", style="o")
plt.ylabel("PRICES")
plt.show()

# Taget-variable name
TargetName = "PRICES"
# Explanatory variables to be used
FeaturesName = [\
              #-- "Crime occurrence rate per unit population by town"
              "CRIM",\
              #-- "Percentage of 25000-squared-feet-area house"
              'ZN',\
              #-- "Percentage of non-retail land area by town"
              'INDUS',\
              #-- "Index for Charlse river: 0 is near, 1 is far"
              'CHAS',\
              #-- "Nitrogen compound concentration"
              'NOX',\
              #-- "Average number of rooms per residence"
              'RM',\
              #-- "Percentage of buildings built before 1940"
              'AGE',\
              #-- 'Weighted distance from five employment centers'
              "DIS",\
              ##-- "Index for easy access to highway"
              'RAD',\
              ##-- "Tax rate per $100,000"
              'TAX',\
              ##-- "Percentage of students and teachers in each town"
              'PTRATIO',\
              ##-- "1000(Bk - 0.63)^2, where Bk is the percentage of Black people"
              'B',\
              ##-- "Percentage of low-class population"
              'LSTAT',\
              ]

x = df[FeaturesName]
y = df[TargetName]
##-- Logarithmic scaling
y_log = np.log(y)

# Standardize the Variables
sscaler = preprocessing.StandardScaler()
sscaler.fit(x)
x_std = sscaler.transform(x)


x_train, x_test, y_train, y_test = train_test_split(x_std, y_log, test_size=0.2, random_state=0)


regressor = LinearRegression()
regressor.fit(X_train, Y_train)

y_pred_train = regressor.predict(x_train)
y_pred_test = regressor.predict(x_test)

# Inverse logarithmic transformation if necessary
y_pred_train, y_pred_test = np.exp(y_pred_train), np.exp(y_pred_test)
y_train, y_test = np.exp(y_train), np.exp(y_test)


plt.figure(figsize=(5, 5), dpi=100)
plt.xlabel("PRICES")
plt.ylabel("Predicted PRICES")
plt.xlim(0, 60)
plt.ylim(0, 60)
plt.scatter(y_train, y_pred_train, lw=1, color="r", label="train data")
plt.scatter(y_test, y_pred_test, lw=1, color="b", label="test data")
plt.legend()
plt.show()


R2 = r2_score(y_test, y_pred_test)
print(f'R2 score: {R2:.2f}')







