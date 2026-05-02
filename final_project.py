#Final Project - Olga Ivashyna
#ECO4443 - Aprul 30th
#Link to initial datasets: https://github.com/fivethirtyeight/data/tree/master/redlining

import os
import numpy as np
import pandas as pd
import sqlite3
import statsmodels.api as sm # Array-based interface
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt

sqlite3.sqlite_version # SQLite version

#importing files - note that these are the edited-down versions
try:
   metro_grades = pd.read_csv("metro-grades.csv")
        
except FileNotFoundError:
    print("Could not load metro-grades.csv. Please check that the file exists.")

print(metro_grades.head())

# Summary info
print(metro_grades.info())

# Descriptive statistics
print(metro_grades.describe(include="all"))

try:
   zone_blocks = pd.read_csv("zone-block-summary.csv")
        
except FileNotFoundError:
    print("Could not load zone-block-summary.csv. Please check that the file exists.")

print(zone_blocks.head())

# Summary info
print(zone_blocks.info())

# Descriptive statistics
print(zone_blocks.describe(include="all"))


redline_full = pd.merge(metro_grades,zone_blocks, on= "metro_area").fillna(0)

def regress(group, y_col, x_cols):
    X = group[x_cols]
    X = sm.add_constant(X) # Adds an intercept
    Y = group[y_col]
    model = sm.OLS(Y, X).fit()
    return model.params  # Returns coefficients for each group


# Example usage
results = df.groupby('category_column').apply(regress, 'target_y', ['feature_x'])

    

