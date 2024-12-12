import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
np.random.seed(42)

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

import os


import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

dataset = pd.read_csv('housing.csv')

# print(dataset.head())

dataset.shape

# Get dataset information
dataset.info()

# Get which categories belong to the ocean_proximity text field
set(dataset['ocean_proximity'])

# Count how many neighborhoods there are in each category
dataset['ocean_proximity'].value_counts()

# We can analise the data with the describe() function
dataset.describe()

# Analise the data with histograms
dataset.hist(bins=50, figsize=(20,15))

# plt.show()

# Divide the bases between train and test
df_train, df_test = train_test_split(dataset, test_size=0.2, random_state=7)

# print(len(df_train), "for training + ", len(df_test), "for testing")

# Divide by 1.5 to limit the number of income categories
# dividing the value of the "median_income" column of each entry by the value 1.5 and then rounding the result up using the
# np.ceil() function (from the NumPy library). This creates a new column called "income_cat" in the dataset that contains the values ​​of the income categories after
# division and rounding.
dataset['income_cat'] = np.ceil(dataset['median_income'] / 1.5) 

# Label those above 5 as 5.
# Values ​​in the "income_cat" column that are greater than or equal to 5 are replaced with 5. This is done using the pandas .where() function.
# Basically, if the value in "income_cat" is less than 5, it remains the same; otherwise, it is replaced with 5.
dataset['income_cat'].where(dataset['income_cat'] < 5, 5.0, inplace=True)

dataset['income_cat'] = pd.cut(dataset['median_income'],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

print(dataset['income_cat'].value_counts())

print(dataset['income_cat'].hist())

# Let's perform stratified sampling based on income category!
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(dataset, dataset['income_cat']):
    strat_train_set = dataset.loc[train_index]
    strat_test_set = dataset.loc[test_index]


print(strat_train_set['income_cat'].value_counts() / len(strat_train_set))
print(strat_test_set['income_cat'].value_counts() / len(strat_test_set))

print(dataset['income_cat'].value_counts() / len(dataset))

# Removing income_cat from the training and testing sets
# The use of the term set_ is a convention to indicate that it is a temporary variable that iterates over a set of data
for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)

# Analysing geographic data
housing = strat_test_set.copy()
# housing.plot(kind='scatter', x='longitude', y='latitude')

# housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)
# plt.show()

housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
             s=housing['population'] / 100, label='population', figsize=(10,7),
             c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True, 
             sharex=False)
plt.legend()
# plt.show()

corr_matrix = housing.corr(numeric_only=True)

print(corr_matrix['median_house_value'].sort_values(ascending=False))

# scatter_matrix
attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
print(scatter_matrix(housing[attributes], figsize=(12, 8)))
# plt.show()

housing.plot(kind='scatter', x='median_income', y='median_house_value',
             alpha=0.1)
plt.axis([0, 16, 0, 550000])
# plt.show()

# Preparing the data to add to the algorithm
housing = strat_train_set.drop('median_house_value', axis=1) # deleting the target for the training base (our X)
housing_labels = strat_train_set['median_house_value'].copy() # saving the target (our Y)

# Listing null columns
sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
print(sample_incomplete_rows)

print(housing.isnull().sum())

# total_bedrooms have 158 NaN values
# We'll calculate the median value for the total_bedrooms and update the column with this value

# Option one - Replace the null values with the median

# median = housing['total_bedrooms'].median()
# sample_incomplete_rows['total_bedrooms'].fillna(median, inplace=True)
# print(sample_incomplete_rows)

# Option two - Use the Sklearn classes
try:
    from sklearn.impute import SimpleImputer # Scikit-learn 0.20+
except ImportError:
    from sklearn.preprocessing import Imputer as SimpleImputer

imputer = SimpleImputer(strategy='median')

# Remove the text attribute because the median can only be calculated on numeric attributes:
housing_num = housing.drop('ocean_proximity', axis=1)
imputer.fit(housing_num) # calculating the median of each attribute and storing the result in the statistics_ variable

imputer.statistics_ # print to see

# compare with the manual calculation 
housing_num.median().values # print to see

# Applying the "trained" Imputer to the base to replace missing values ​​lost by the median:
X = imputer.transform(housing_num) # the result is an array
# print(X)

# Transform it to a dataframe again
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
# print(housing_tr)

# check the results
housing_tr.loc[sample_incomplete_rows.index.values]

# check the strategy used 
print(imputer.strategy)

print(housing_tr.head())

# Now let's preprocess the categorical input feature, "ocean_proximity"
housing_cat = housing[['ocean_proximity']]
print(housing_cat.head(10))

# OrdinalEncoder is a class from the scikit-learn library, 
# used to transform ordinal categorical variables into numeric values.
try:
    from sklearn.preprocessing import OrdinalEncoder
except ImportError:
    from future_encoders import OrdinalEncoder # Scikit-Learn < 0.20

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print(housing_cat_encoded)

print(ordinal_encoder.categories_)

# OneHotEncoder is another class from the scikit-learn library, 
# used to transform categorical variables into binary numeric representations.
try:
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.preprocessing import OneHotEncoder
except ImportError:
    from future_encoders import OneHotEncoder # Scikit-Learn < 0.20

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot

print(cat_encoder.categories_)

# Creating pipeline of the data pre-processing 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')), # replacing null values with the median
    ('std_scalar', StandardScaler()) # standardizing data scales
])

housing_num_tr = num_pipeline.fit_transform(housing_num)

print(housing_num_tr)

# Now, let's deal with categorical values:
try:
    from sklearn.compose import ColumnTransformer
except ImportError:
    from future_encoders import ColumnTransformer # Scikit-learn < 0.20

num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs), # handling numeric variables (calling the pipeline from above)
    ('cat', OneHotEncoder(), cat_attribs), # handling categorical variables
])

housing_prepared = full_pipeline.fit_transform(housing)
# print(housing_prepared)
# print(housing_prepared.shape)
# print(type(housing_prepared))

# Notice that the result is a multidimensional array. We need to transform it to a dataframe
column_names = [
    'longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms',
    'population', 'households', 'median_income', '<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']

housing_df = pd.DataFrame(data=housing_prepared, columns=column_names)
print(housing_df.head())

# No more null values 
print(housing_df.isnull().sum())

# Choosing the best regression model
# Linear Regression? Let's check that
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]

# print(some_data)
# print(some_labels)

some_data_prepared = full_pipeline.transform(some_data)
predictions = lin_reg.predict(housing_prepared)

print("Predictions:", lin_reg.predict(some_data_prepared))

# Compare with the real values
print("Labels:", list(some_labels))

# Analysing the model

# MSE measures the mean square of the differences between the values 
# ​​predicted by the model and the actual values ​​observed in the data set.
# The lower the MSE value, the better the model fits the data.
from sklearn.metrics import mean_squared_error
housing_predictions = predictions
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse) # 69050.56

# Another metric
from sklearn.metrics import mean_absolute_error
lin_mae = mean_absolute_error(housing_labels, predictions)
print(lin_mae) # 49905.32

# Let's try another one
from sklearn.metrics import r2_score
r2 = r2_score(housing_labels, housing_predictions)
print('r²',r2) # 0.64

# Function to calculate the MAPE (Mean Absolute Percentage Error)
def calculate_mape(labels, predictions):
    errors = np.abs(labels - predictions)
    print('errors', errors)
    relative_errors = errors / np.abs(labels)
    mape = np.mean(relative_errors) * 100
    return mape

# Calculate MAPE
mape_result = calculate_mape(housing_labels, housing_predictions)

print(f"O MAPE é: {mape_result:.2f}%") # 28.65%

# Let's try a tree regression model
from sklearn.tree import DecisionTreeRegressor

model_dtr = DecisionTreeRegressor(max_depth=10)
model_dtr.fit(housing_prepared, housing_labels)

# Let's repeat the same process as we did with the LinearRegression
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
predictions = model_dtr.predict(some_data_prepared)

print("Predictions:", model_dtr.predict(some_data_prepared))
# Compare with the real values
print("Labels:", list(some_labels))

# mean_squared_error
housing_predictions = model_dtr.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)

# mean_absolute_error
lin_mae = mean_absolute_error(housing_labels, housing_predictions)
print(lin_mae)

r2 = r2_score(housing_labels, housing_predictions)
print('r²',r2)

# Calculate the MAPE
mape_result = calculate_mape(housing_labels, housing_predictions)

print(f"O MAPE é: {mape_result:.2f}%")

# FINAL CONCLUSION 
# The Tree regression model perfomed much better than the linear regression model.
# MAPE = 28% against 17.94%
# There are other models that we can try or even improve these ones