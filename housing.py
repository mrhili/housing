# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 23:02:27 2019

@author: HP
"""

import os
import tarfile
from six.moves import urllib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
#Custom Transformers
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

#handle Pandas DataFrames,
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

#from stack overflow
class SupervisionFriendlyLabelBinarizer(LabelBinarizer):
    def fit_transform(self, X, y=None):
        return super(SupervisionFriendlyLabelBinarizer, self).fit_transform(X)


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    
    
    
def load_housing_data():
    csv_path = "housing.csv"
    return pd.read_csv(csv_path)

housing = load_housing_data()

#print(housing.head())

#print(housing.info())


#print(housing["ocean_proximity"].value_counts()  )

'''
print(housing.describe())


housing.hist(bins=50, figsize=(20,15))
plt.show()
'''

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
#print(len(train_set), "train +", len(test_set), "test")


housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

'''
housing.hist(bins=50, figsize=(20,15))
plt.show()

'''
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    
'''
3.0    0.350581
2.0    0.318847
4.0    0.176308
5.0    0.114438
1.0    0.039826
Name: income_cat, dtype: float64
'''
#print( housing["income_cat"].value_counts() / len(housing) )


#removing the income_cat
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
    
#JUST exploring the data
#copiying without harmfulling the data
housing = strat_train_set.copy()

#housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"]/100, label="population", figsize=(10,7), c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True) 
plt.legend()
#correlation
corr_matrix = housing.corr() 

corr_matrix["median_house_value"].sort_values(ascending=False)

print( corr_matrix)


attributes = ["median_house_value", "median_income", "total_rooms",
"housing_median_age"]
#scatter_matrix(housing[attributes], figsize=(12, 8))

housing.plot(kind="scatter", x="median_income", y="median_house_value",
alpha=0.1)


'''

BEGIN Preparing the Data for Machine Learning Algorithms

**********************************************************************************

'''
print('*************************Preparing the Data for Machine Learning Algorithms*********************************')
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

#print(housing)

#print(housing_labels)

#Taking care the missings data
'''
housing.dropna(subset=["total_bedrooms"]) # option 1
housing.drop("total_bedrooms", axis=1) # option 2

median = housing["total_bedrooms"].median() # option 3
housing["total_bedrooms"].fillna(median, inplace=True)
'''


imputer = Imputer(strategy="median")

housing_num = housing.drop("ocean_proximity", axis=1)


imputer.fit(housing_num)
'''
print(imputer.statistics_)
[-118.51     34.26     29.     2119.5     433.     1164.      408.
    3.5409]

print(housing_num.median())
longitude             -118.5100
latitude                34.2600
housing_median_age      29.0000
total_rooms           2119.5000
total_bedrooms         433.0000
population            1164.0000
households             408.0000
median_income            3.540
'''
X = imputer.transform(housing_num)


housing_tr = pd.DataFrame(X, columns=housing_num.columns)

print('*************************Handling Text and Categorical Attributes*********************************')

encoder = LabelEncoder()


housing_cat = housing["ocean_proximity"]

housing_cat_encoded = encoder.fit_transform(housing_cat)
'''
print(housing_cat_encoded)
[0 0 4 ... 1 0 3]
'''

'''
print(encoder.classes_)
['<1H OCEAN' 'INLAND' 'ISLAND' 'NEAR BAY' 'NEAR OCEAN']
'''
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
'''
print(housing_cat_1hot)
  (0, 0)        1.0
  (1, 0)        1.0
  (2, 4)        1.0
  (3, 1)        1.0
  (4, 0)        1.0
  (5, 1)        1.0
  (6, 0)        1.0
  (7, 1)        1.0
  (8, 0)        1.0
  (9, 0)        1.0
  (10, 1)       1.0
  (11, 1)       1.0
  (12, 0)       1.0
  (13, 1)       1.0
  (14, 1)       1.0
  (15, 0)       1.0
  (16, 3)       1.0
  (17, 1)       1.0
  (18, 1)       1.0
  (19, 1)       1.0
  (20, 0)       1.0
  (21, 0)       1.0
  (22, 0)       1.0
  (23, 1)       1.0
  (24, 1)       1.0
  :     :
  (16487, 1)    1.0
  (16488, 1)    1.0
  (16489, 4)    1.0
  (16490, 3)    1.0
  (16491, 0)    1.0
  (16492, 3)    1.0
  (16493, 1)    1.0
  (16494, 1)    1.0
  (16495, 0)    1.0
  (16496, 1)    1.0
  (16497, 3)    1.0
  (16498, 1)    1.0
  (16499, 0)    1.0
  (16500, 0)    1.0
  (16501, 0)    1.0
  (16502, 4)    1.0
  (16503, 0)    1.0
  (16504, 1)    1.0
  (16505, 1)    1.0
  (16506, 0)    1.0
  (16507, 1)    1.0
  (16508, 1)    1.0
  (16509, 1)    1.0
  (16510, 0)    1.0
  (16511, 3)    1.0
'''

'''
housing_cat_1hot.toarray()
array([[ 1., 0., 0., 0., 0.],
[ 1., 0., 0., 0., 0.],
[ 0., 0., 0., 0., 1.],
...,
[ 0., 1., 0., 0., 0.],
[ 1., 0., 0., 0., 0.],
[ 0., 0., 0., 1., 0.]])
'''


encoder = SupervisionFriendlyLabelBinarizer()
housing_cat_1hot = encoder.fit_transform(housing_cat)


'''
print(housing_cat_1hot)

[[1 0 0 0 0]
 [1 0 0 0 0]
 [0 0 0 0 1]
 ...
 [0 1 0 0 0]
 [1 0 0 0 0]
 [0 0 0 1 0]]

'''


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)




print('*************************Feature Scaling && Transformation Pipelines*********************************')





num_pipeline = Pipeline([ ('imputer', Imputer(strategy="median")),('attribs_adder', CombinedAttributesAdder()),('std_scaler', StandardScaler()),])
housing_num_tr = num_pipeline.fit_transform(housing_num)



num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
num_pipeline = Pipeline([('selector', DataFrameSelector(num_attribs)),('imputer', Imputer(strategy="median")),('attribs_adder', CombinedAttributesAdder()),('std_scaler', StandardScaler()),])
cat_pipeline = Pipeline([('selector', DataFrameSelector(cat_attribs)),('label_binarizer', SupervisionFriendlyLabelBinarizer()),])


full_pipeline = FeatureUnion(transformer_list=[("num_pipeline", num_pipeline),("cat_pipeline", cat_pipeline),])


housing_prepared = full_pipeline.fit_transform(housing)

'''
print(housing_prepared)
[[-1.15604281  0.77194962  0.74333089 ...  0.          0.
   0.        ]
 [-1.17602483  0.6596948  -1.1653172  ...  0.          0.
   0.        ]
 [ 1.18684903 -1.34218285  0.18664186 ...  0.          0.
   1.        ]
 ...
 [ 1.58648943 -0.72478134 -1.56295222 ...  0.          0.
   0.        ]
 [ 0.78221312 -0.85106801  0.18664186 ...  0.          0.
   0.        ]
 [-1.43579109  0.99645926  1.85670895 ...  0.          1.
   0.        ]]
'''



print('*************************Modeling and predections*********************************')


'''

MODEL SAVING LOADING
joblib.dump(my_model, "my_model.pkl")
# and later...
my_model_loaded = joblib.load("my_model.pkl")

'''

#lin_reg = LinearRegression()
#tree_reg = DecisionTreeRegressor()
forest_reg = RandomForestRegressor()

#lin_reg.fit(housing_prepared, housing_labels)
#tree_reg.fit(housing_prepared, housing_labels)

#uncomment to simply lunch the model
#forest_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

'''
print("Predictions:", lin_reg.predict(some_data_prepared))
Predictions: [210644.60459286 317768.80697211 210956.43331178  59218.98886849
 189747.55849879]
print("Labels:", list(some_labels))
Labels: [286600.0, 340600.0, 196900.0, 46300.0, 254500.0]
'''

#housing_predictions = lin_reg.predict(housing_prepared)
#housing_predictions = tree_reg.predict(housing_prepared)


'''
#uncomment to simply lunch the model
housing_predictions = tree_reg.predict(housing_prepared)


mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)
print(rmse)
'''


print('*************************ِCross validation*********************************')

'''
#uncomment to simply lunch the model
scores = cross_val_score(forest_reg, housing_prepared, housing_labels,scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

display_scores(rmse_scores)
'''

print('************************Fining model************************')

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]



grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')

grid_search.fit(housing_prepared, housing_labels)


print( 'best param: ', grid_search.best_params_ )

print('best estimator', grid_search.best_estimator_)

'''
RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features=6, max_leaf_nodes=None, min_impurity_decrease=0.0,
           min_impurity_split=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=30, n_jobs=None, oob_score=False,
           random_state=None, verbose=0, warm_start=False)
'''

    
'''
cvres = grid_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
    
    
    
    
    
63825.0479302 {'max_features': 2, 'n_estimators': 3}
55643.8429091 {'max_features': 2, 'n_estimators': 10}
53380.6566859 {'max_features': 2, 'n_estimators': 30}
60959.1388585 {'max_features': 4, 'n_estimators': 3}
52740.5841667 {'max_features': 4, 'n_estimators': 10}
50374.1421461 {'max_features': 4, 'n_estimators': 30}
58661.2866462 {'max_features': 6, 'n_estimators': 3}
52009.9739798 {'max_features': 6, 'n_estimators': 10}
50154.1177737 {'max_features': 6, 'n_estimators': 30}
57865.3616801 {'max_features': 8, 'n_estimators': 3}
51730.0755087 {'max_features': 8, 'n_estimators': 10}
49694.8514333 {'max_features': 8, 'n_estimators': 30}
62874.4073931 {'max_features': 2, 'n_estimators': 3, 'bootstrap': False}
54561.9398157 {'max_features': 2, 'n_estimators': 10, 'bootstrap': False}
59416.6463145 {'max_features': 3, 'n_estimators': 3, 'bootstrap': False}
52660.245911 {'max_features': 3, 'n_estimators': 10, 'bootstrap': False}
57490.0168279 {'max_features': 4, 'n_estimators': 3, 'bootstrap': False}
51093.9059428 {'max_features': 4, 'n_estimators': 10, 'bootstrap': False}
'''




print('************************Analyze the Best Models and Their Errors************************')

feature_importances = grid_search.best_estimator_.feature_importances_



'''
print( feature_importances )




array([ 7.33442355e-02, 6.29090705e-02, 4.11437985e-02,
1.46726854e-02, 1.41064835e-02, 1.48742809e-02,
1.42575993e-02, 3.66158981e-01, 5.64191792e-02,
1.08792957e-01, 5.33510773e-02, 1.03114883e-02,
1.64780994e-01, 6.02803867e-05, 1.96041560e-03,
2.85647464e-03])
'''



extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_one_hot_attribs = list(encoder.classes_)
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
print( sorted(zip(feature_importances, attributes), reverse=True)   )
'''
[(0.3649872065292531, 'median_income'), (0.16345995297979102, 'INLAND'), (0.10871745506802107, 'pop_per_hhold'), (0.07106602726229345, 'longitude'), (0.06348187748517971, 'latitude'), (0.05976377472126414, 'rooms_per_hhold'), (0.055226504029559115, 'bedrooms_per_room'), (0.04160317729033542, 'housing_median_age'), (0.015602835179929796, 'total_rooms'), (0.01556973296793557, 'total_bedrooms'), (0.015555762309639353, 'population'), (0.01412105838060941, 'households'), (0.005744299195428113, '<1H OCEAN'), (0.0033041157087794963, 'NEAR OCEAN'), (0.0017326053747885087, 'NEAR BAY'), (6.361551719267063e-05, 'ISLAND')]
'''



print('*****************Evaluation************************')

final_model = grid_search.best_estimator_


X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse) # => evaluates to 47,766.0   /// 47572.549732424035




