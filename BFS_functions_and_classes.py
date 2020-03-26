from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


def get_percentages(df: pd.DataFrame, feature: str) -> list:
    """Function for getting the percentages of all categories of a discrete feature in a dataframe."""
    prc = df.groupby(feature).count().iloc[:, 0]
    keys = prc.keys()
    prc = [prc[x]/df[feature].count() for x in keys]
    for i in range(0, len(keys)):
        print(f"{keys[i]} -> {round(prc[i]*100,2)}%")
    return prc


def stacked_bar_plot(df: pd.DataFrame, x: str, y: str, percent=True):
    """Function for generating a stacked bar plot using two discrete features in a data frame.
    If percent=True, all columns in the will have an equal length, allowing for easier visualization
    of the difference in composition between the categories.
    If percent=False, the length of each column will be based on the prevalence of its corresponding
    category in the data set.
    """
    total = df.groupby(x)[y].count()
    xkeys = total.keys()
    ykeys = sorted(df[y].unique())
    # Generate the first part of every column in the plot. That part corresponds to the
    # first category of the second feature(y)
    values = df.loc[df[y] == ykeys[0]].groupby(x)[y].count()
    if percent:
        values = [values[z] / total[z] for z in xkeys]
    plt.bar(xkeys, values, width=0.8, label=ykeys[0])
    # Generate the rest of the parts in the columns by iterating through the remaining
    # categories in y
    for i in range(1, len(ykeys)):
        values_new = df.loc[df[y] == ykeys[i]].groupby(x)[y].count()
        if percent:
            values_new = [values_new[z] / total[z] for z in xkeys]
        plt.bar(xkeys, values_new, width=0.8, label=ykeys[i], bottom=values)
        values = values_new

    plt.xticks(xkeys)
    plt.xlabel(x)
    plt.ylabel(y)
    if percent:
        plt.ylim(top=1.2)
    plt.legend(fontsize=13)
    plt.title(f'Distribution of {y} over {x} in the data set', fontsize=16, y=1.1)


def get_strong_preds(df: pd.DataFrame, feature: str, print_vals=False):
    """Function that finds and returns the categories in a discrete feature which
    have a significant effect on the dependent variable (Purchase)
    If print_vals=True, the function will print the kept categories and their p values.
    Else, the function will return a list containing the kept categories.
    """
    kept = []
    for key in sorted(df[feature].unique()):
        one = df['Purchase_normal'].loc[df[feature] == key]
        rest = df['Purchase_normal'].loc[df[feature] != key]
        # The function runs t-tests between the categories which yield a probability
        # value representing how likely it is that there is no significant difference
        # between the categories. This function only returns the categories with a
        # probability value smaller than 5%
        p_val = stats.ttest_ind(one, rest)[1]
        if not print_vals:
            if p_val < 0.05:
                kept.append(key)
        else: 
            print(key, " ", round(p_val, 3))
    if not print_vals:
        return kept


class GroupByCustomer(BaseEstimator, TransformerMixin):
    """Custom Transformer for grouping the initial data frame based on the user ID.
    The function aggregates the features using their mode.
    The function also applies a LabelEncoder on the Gender column of the data frame.
    ----------
    Attributes
    ----------
    features : tuple
    - the features to be aggregated and kept in the new DataFrame -
    """

    def __init__(self, features=('Gender', 'Age', 'Occupation', 'City_Category',
                                 'Stay_In_Current_City_Years', 'Marital_Status')):
        self.encoder = LabelEncoder()
        self.features = features

    def fit(self, X: pd.DataFrame):
        self.encoder.fit(X['Gender'])
        return self

    def transform(self, X: pd.DataFrame):
        X = X.groupby('User_ID')[self.features].agg(pd.Series.mode)
        X['Gender'] = self.encoder.transform(X['Gender'])
        return X


class AttributesAdder(BaseEstimator, TransformerMixin):
    """Custom Transformer that applies OneHotEncoding on discrete features of
    a DataFrame, then applies t-tests on the categories, keeping the ones with a
    significant effect on the given dependent variable, y.
    ----------
    Attributes
    ----------
    target : pd.DataFrame
    - the target variable used to compute the difference between the categories
    and the p values for their t tests -
    features : tuple
    - the features in the DataFrame AttributesAdder will be applied to -
    """

    def __init__(self, target: pd.DataFrame, features=('Occupation', 'Age', 'City_Category', 'Stay_In_Current_City_Years')):
        self.features = features
        # self.encoders will hold the fitted OneHotEncoders for all the features in self.features.
        self.encoders = dict()
        self.y = target
        
    def get_strong_preds(self, X: pd.DataFrame, feature: str) -> list:
        """Method that finds and returns the categories in a discrete feature which
            have a significant effect on the dependent variable.
            Similar to the get_strong_preds function, but the dependent variable needs to
            be specified and it doesn't have the option to print the list of kept anymore.
            """
        kept = []
        for i in sorted(X[feature].unique()):
            one = self.y.loc[X[feature] == i]
            rest = self.y.loc[X[feature] != i]
            p_val = stats.ttest_ind(one, rest)[1]
            if p_val < 0.05:
                kept.append(i)
        return kept
    
    def add_preds(self, X: pd.DataFrame, feature: str):
        """Function for fitting the OneHotEncoders to the features and saving them
        inside the self.encoders dict, as well as all the categories of the features
        and the categories with a significant effect on the dependent variable. """
        kept = self.get_strong_preds(X, feature)
        keys = sorted(X[feature].unique())
        # this is to ensure that if all elements have a statistically significant p value the data
        # frame still does not retain every single category as the last one would be redundant
        if len(kept) == len(keys):
            kept.pop()
        encoder = OneHotEncoder(sparse=False)
        encoder.fit(np.array(X[feature]).reshape(-1, 1))
        self.encoders[feature] = [keys, kept, encoder]

    def fit(self, X: pd.DataFrame):
        for feature in self.features:
            self.add_preds(X, feature)
        return self

    def transform(self, X: pd.DataFrame):
        for feature in self.features:
            # Applies the OneHotEncoders' transform on their corresponding features.
            encoded = self.encoders[feature][2].transform(np.array(X[feature]).reshape(-1, 1))
            encoded = pd.DataFrame(encoded, columns=self.encoders[feature][0])
            # Iterates through all categories in the feature and checks if they are in the kept list.
            # If a category is in kept, a new binary feature is added to represent the category.
            for key in self.encoders[feature][0]:
                if key in self.encoders[feature][1]:
                    X[f'{feature} {key}'] = list(encoded[key])
            X = X.drop(feature, axis=1)
        
        return X
