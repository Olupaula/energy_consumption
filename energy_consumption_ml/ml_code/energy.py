import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_validate

from sklearn.base import BaseEstimator

from sklearn.pipeline import Pipeline

from sklearn.linear_model import (
    LinearRegression,
    SGDRegressor,
    BayesianRidge,
)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.feature_selection import RFE

from sklearn.metrics import make_scorer, mean_squared_error, r2_score

import joblib

data = pd.read_csv("/Users/user/PycharmProjects/"
                   + "regression_ml/energy_consumption/energy_consumption_ml/data/Steel_industry_data.csv")

# Taking a picture of the data
print(data.head())

# Viewing the type of each column
print(data.dtypes)

features = data.drop(labels=["date"], axis=1).columns

numerical_features = [
    feature for feature in features
    if data[feature].dtype == np.float64
    or data[feature].dtype == np.int64
]

categorical_features = [
   feature for feature in features if feature not in numerical_features
]

# Descriptive Statistics
plots = dict()
for feature in numerical_features:
    if feature != 'Usage_kWh':
        plot = sns.scatterplot(x=data[feature], y=data['Usage_kWh'])
        feature = feature.replace('_', ' ').replace('.', ' ').replace('(t', ' (t')

        plot.set(
            title="Plot of " + feature + ' against Electric Energy Consumption (Usage) in Kwh',
            xlabel=plot.get_xlabel().replace('_', ' ').replace('.', ' ').replace('(t', ' (t'),
            ylabel=plot.get_ylabel().replace('_', ' ').replace('.', ' ').replace('(t', ' (t')
        )

        plt.savefig(
                '/Users/user/PycharmProjects/regression_ml/'
                + 'energy_consumption/energy_consumption_images/plot_of_' + feature + '_against_electric_energy_'
                + 'consumption.png'
        )
        plt.show()


# Viewing the correlation between the independent variables to take care of collinearity
print("The correlations")
pd.options.display.max_columns = 12
print(data.iloc[:, 1:7].corr())
# It can be seen that the 'CO2(tCO2)' and  'Lagging_Current_Reactive.Power_kVarh' are highly correlated (0.8870)
# The feature 'Leading_Current_Power_Factor' and 'Leading_Current_Reactive_Power_kVarh'
# are also highly correlated (-0.9441). Hence, only one feature will be selected from each pair. In this case, I choose
# 'CO2(tCO2)' and 'Leading_Current_Power_Factor'. The other variables will be dropped

data = data.drop(labels=['Leading_Current_Reactive_Power_kVarh', 'Lagging_Current_Reactive.Power_kVarh'], axis=1)

print("New Data")

# Viewing the correlations again
print(data.iloc[:, 1:6].corr())
# The data is now good to be used for regression

# The correlation of the other variables with power usage (Usage_kWh) alone
print(data.iloc[:, 1:6].corr().iloc[:, 0])

index_of_non_numeric = []  # A list of indices of non-numeric features
index_of_numeric = []  # A list of indices of numeric features

for col in range(len(data.columns)):
    if data.iloc[:, col].dtype != np.float64 and data.iloc[:, col].dtype != np.int64:
        index_of_non_numeric.append(col)
    else:
        index_of_numeric.append(col)

cat_data = data.iloc[:, index_of_non_numeric]
print(cat_data)

# The unique values
features_unique_values = cat_data.apply(lambda x: x.unique())
print(features_unique_values)

# The number of unique values, specifically obtained because of the dates
no_of_unique_values = cat_data.apply(lambda x: len(x.unique()))
print(no_of_unique_values)

# Visualizing the relationship between the target variable and the categorical variables
for feature in categorical_features:
    if feature != 'date':
        plot = sns.stripplot(x=data[feature], y=data.iloc[:, 1])
        feature = feature.replace('_', ' ').replace('.', ' ').replace('(t', ' (t').replace(
            'week',
            'Week'
        ).replace('WeekStatus', 'Week Status')

        plot.set(
            title="Plot of " + feature + ' against Electric Energy Consumption (Usage) in Kwh',
            xlabel=plot.get_xlabel().replace('_', ' ').replace('.', ' ').replace('(t', ' (t').replace(
                'week',
                'Week'
            ).replace('WeekStatus', 'Week Status'),
            ylabel=plot.get_ylabel().replace('_', ' ').replace('.', ' ').replace('(t', ' (t').replace(
                'week',
                'Week'
            ).replace('WeekStatus', 'Week Status')
        )

        plt.savefig(
            '/Users/user/PycharmProjects/regression_ml/'
            + 'energy_consumption/energy_consumption_images/plot_of_' + feature + '_against_electric_energy_'
            + 'consumption.png'
        )
        plt.show()


# recoding the categorical variables
# Looking  at the stripplots, for the categorical variables, it makes sense to recode the variables thus
data.WeekStatus = data.WeekStatus.apply(
    lambda w:
    0 if w == "Weekday"
    else 1  # Weekend
)

data.Day_of_week = data.Day_of_week.apply(
    lambda d:
    0 if d == "Sunday"  # because the power usage is smaller on Sundays than on other days
    else 1
)

data.Load_Type = data.Load_Type.apply(
    lambda l:
    0 if l == 'Light_Load'  # because the light_load uses less power than the other two load_types
    else 1
)

y = data.iloc[:, 1]
x = data.drop(labels=['Usage_kWh', 'date'], axis=1)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=456)


# A class that handles model fitting, prediction and scoring of estimators
class RegressorSwitcher(BaseEstimator):
    def __init__(self, estimator=SGDRegressor()):
        """
         A custom BaseEstimator that can switch between regressors.
         :param estimator: sklearn object - The regressor
        """
        self.estimator = estimator

    def fit(self, X, y=None, **kwargs):

        self.estimator.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.estimator.predict(X)

    def score(self, X, y):
        return self.estimator.score(X, y)


pipeline = Pipeline([
    ('feature_selection', RFE(estimator=LinearRegression(), n_features_to_select=5, step=1)),
    ('model', RegressorSwitcher())
])

print('step 1', pipeline.steps[0])
print('step 2', pipeline.steps[1])

index = list(range(1, 19, 2))
print(index)

kNeighbors = [KNeighborsRegressor(n_neighbors=i) for i in index]
print(kNeighbors)

dt_indices = ((2, 2, 2), (2, 4, 10), (2, 6, 20))
decisionTree = [DecisionTreeRegressor(max_depth=md, max_features=mf, max_leaf_nodes=mln) for md, mf, mln in dt_indices]
parameters = [
    {
        'model__estimator': [LinearRegression()]
    },

    {
        'model__estimator': kNeighbors,
     },

    {
        'model__estimator': decisionTree
    },

    {
        'model__estimator': [BayesianRidge(
            alpha_1=0.2,
            alpha_2=0.2,
            lambda_1=0.2,
            lambda_2=0.2,
            alpha_init=None,
            lambda_init=None,
        )]
    }
]

scoring = {'mean_squared_error': make_scorer(mean_squared_error)}
grid_search_models = GridSearchCV(
    pipeline,
    param_grid=parameters,
    cv=5,
    scoring={'r2': make_scorer(r2_score), 'mse': make_scorer(mean_squared_error)},
    refit='r2'
)

grid_search_models.fit(x_train, y_train)
print('best_score=', grid_search_models.best_score_)
print('The best_estimator is', grid_search_models.best_params_)
p = grid_search_models.score(x_test, y_test)
print('test_score', p)

# The selected features
truths_of_selection = grid_search_models.best_estimator_.named_steps['feature_selection'].get_support()
selected_features = x.columns[truths_of_selection]
print('features selected =', selected_features)

# Determining the best model
print(grid_search_models.cv_results_['mean_test_mse'])
print(grid_search_models.cv_results_['mean_test_r2'])
print(grid_search_models.cv_results_)
# print(grid_search_models.cv_results_['mean_test_score'])
# It can thus be seen that the Linear Regression has the best

# A Data Frame that summarizes the result
result = pd.DataFrame()
result["Model Name"] = grid_search_models.cv_results_['param_model__estimator']
result["Coefficient of Determination"] = grid_search_models.cv_results_['mean_test_r2'].round(decimals=4)
result["Mean Squared Error"] = grid_search_models.cv_results_['mean_test_mse'].round(decimals=4)

print(result)

# Reducing the x columns to the number of selected features
x_train = x_train[x_train.columns[truths_of_selection]]
x_test = x_test[x_test.columns[truths_of_selection]]
# print(x_train.columns)
# print(x_test.columns)

# Fitting the Best Model according to GridSearchCV
model = LinearRegression()
model.fit(x_train, y_train)

cross_validation = cross_validate(
    model,
    x_train,
    y_train,
    scoring={
        'mse': make_scorer(mean_squared_error),
        'r2': make_scorer(r2_score)
    },
    cv=5)

r2_score = cross_validation['test_r2'].mean().round(decimals=4)
mse = cross_validation['test_mse'].mean().round(decimals=4)

test_score = model.score(x_test, y_test)

# Verifying metrics
print('Average Cross validation Coefficient of Determination (r2 score) =', r2_score.mean().round(decimals=4))
print('Average Cross validation Mean Squared Error (MSE) =', mse.mean().round(decimals=4))
print('Test score =', round(test_score, 4))
# The metrics remain the same

# Saving the best model
joblib_file = model
joblib.dump(
    joblib_file,
    '/Users/user/PycharmProjects/regression_ml/energy_consumption/'
    + 'energy_consumption_ml/model/LinearRegression.joblib'
)

# Loading the saved model
model = joblib.load(
    '/Users/user/PycharmProjects/regression_ml/energy_consumption/'
    + 'energy_consumption_ml/model/LinearRegression.joblib'
)

# Evaluation with Test Data
print('Test Scores')
print('predicted = ', round(model.score(x_test, y_test), 4))
print('mse = ', round(mean_squared_error(y_test, model.predict(x_test)), 4))
