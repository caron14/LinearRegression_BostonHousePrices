import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
dataset = load_boston()
sns.set()

f = pd.DataFrame(dataset.data)
f.columns = dataset.feature_names
f["PRICES"] = dataset.target

# Show the relation, "PRICES" vs each variable
f.plot(x="TAX", y="PRICES", style="o")
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

X = f[FeaturesName]
Y = f[TargetName]
##-- Logarithmic scaling
Y_log = np.log(Y)

# Standardize the Variables
from sklearn import preprocessing
sscaler = preprocessing.StandardScaler()
sscaler.fit(X)
X_std = sscaler.transform(X)


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_std, Y_log, test_size=0.2, random_state=99)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

y_pred_train = regressor.predict(X_train)
y_pred_test = regressor.predict(X_test)

# Inverse logarithmic transformation if necessary
y_pred_train, y_pred_test = np.exp(y_pred_train), np.exp(y_pred_test)
Y_train, Y_test = np.exp(Y_train), np.exp(Y_test)


plt.figure(figsize=(5, 5), dpi=100)
plt.xlabel("PRICES")
plt.ylabel("Predicted PRICES")
plt.xlim(0, 60)
plt.ylim(0, 60)
plt.scatter(Y_train, y_pred_train, lw=1, color="r", label="train data")
plt.scatter(Y_test, y_pred_test, lw=1, color="b", label="test data")
plt.legend()
plt.show()


R2 = r2_score(Y_test, y_pred_test)
print(f'R2 score: {R2:.2f}')







