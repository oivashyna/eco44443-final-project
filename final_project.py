#Final Project - Olga Ivashyna
#ECO4443 - Aprul 30th
#Link to initial datasets: https://github.com/fivethirtyeight/data/tree/master/redlining

import os
import numpy as np
import pandas as pd
import sqlite3
import statsmodels.api as sm # Array-based interface
import statsmodels.formula.api as smf
import matplotlib as mpl
import matplotlib.pyplot as plt

#import seaborn as sns
#import matplotlib.pyplot as plt

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

#combine the two dataframes into one
redline_full = pd.merge(metro_grades,zone_blocks, on= "metro_area").fillna(0)

print(redline_full.head())

# Summary info
print(redline_full.info())

# Descriptive statistics
print(redline_full.describe(include="all"))

#correlations
#correlation between percentage and HOLC grade - might change to make it specific grades
for grade in ["A", "B", "C", "D"]:
    indicator = (redline_full["holc_grade_x"] == grade).astype(int)
    
    print(f"\n--- HOLC Grade {grade} ---")
    
    print("pct_white:", redline_full["pct_white"].corr(indicator))
    print("pct_black:", redline_full["pct_black"].corr(indicator))
    print("pct_hisp:", redline_full["pct_hisp"].corr(indicator))
    print("pct_asian:", redline_full["pct_asian"].corr(indicator))
    print("pct_other:", redline_full["pct_other"].corr(indicator))
    
    print("avg_G:", redline_full["avg_G"].corr(indicator))
    print("median_G:", redline_full["median_G"].corr(indicator))
    print("decile_10_G:", redline_full["decile_10_G"].corr(indicator))
    print("quartile_25_G:", redline_full["quartile_25_G"].corr(indicator))


#correlation between percentage and percentage in surrounding areas
pct_cols = ["pct_white", "pct_black", "pct_hisp", "pct_asian", "pct_other"]
surr_cols = [
    "surr_area_pct_white",
    "surr_area_pct_black",
    "surr_area_pct_hisp",
    "surr_area_pct_asian",
    "surr_area_pct_other"
]

for surr in surr_cols:
    print(f"\n--- Correlations with {surr} ---")
    
    for pct in pct_cols:
        corr_val = redline_full[pct].corr(redline_full[surr])
        print(f"{pct} vs {surr}: {corr_val}")


#correlation between 2020 percent change and  HOLC percentage
g_cols = ["avg_G", "median_G", "decile_10_G", "quartile_25_G"]
pct_cols = ["pct_white", "pct_black", "pct_hisp", "pct_asian", "pct_other"]
surr_cols = [
    "surr_area_pct_white",
    "surr_area_pct_black",
    "surr_area_pct_hisp",
    "surr_area_pct_asian",
    "surr_area_pct_other"
]

# Correlation with local percentages
print("\n--- Correlation with local percentages ---")
for g in g_cols:
    for pct in pct_cols:
        corr_val = redline_full[g].corr(redline_full[pct])
        print(f"{g} vs {pct}: {corr_val}")

# Correlation with surrounding area percentages
print("\n--- Correlation with surrounding area percentages ---")
for g in g_cols:
    for surr in surr_cols:
        corr_val = redline_full[g].corr(redline_full[surr])
        print(f"{g} vs {surr}: {corr_val}")

#Simple regression
X3 = pd.DataFrame(redline_full["holc_grade_X"])
X3["homeownership"] = X3["homeownership"].map({"A": 4, "B": 3, "C": 2, "D": 1})
X3 = pd.get_dummies(X3, drop_first=True)
X3 = X3.apply(pd.to_numeric, errors="coerce")
X3 = sm.add_constant(X3)
yw = pd.DataFrame(redline_full["pct_white"])
yb = pd.DataFrame(redline_full["pct_black"])
yh = pd.DataFrame(redline_full["pct_hisp"])
ya = pd.DataFrame(redline_full["pct_asian"])
yo = pd.DataFrame(redline_full["pct_other"])
y = np.c_(yw, yb, yh, ya, yo)

x2w= pd.DataFrame(redline_full["surr_area_pct_white"])
x2b= pd.DataFrame(redline_full["surr_area_pct_black"])
x2h= pd.DataFrame(redline_full["surr_area_pct_hisp"])
x2a= pd.DataFrame(redline_full["surr_area_pct_asian"])
x2o= pd.DataFrame(redline_full["surr_area_pct_other"])
x_surr = np.c_(x2w, x2b, x2h, x2a, x2o)
x_surr = sm.add_constant(x_surr)

xavg = pd.DataFrame(redline_full["avg_G"])
xmed = pd.DataFrame(redline_full["median_G"])
xten = pd.DataFrame(redline_full["decile_10_G"])
xquart = pd.DataFrame(redline_full["quartile_25_G"])
xperc = np.c_(xavg, xmed, xten, xquart)
xperc = sm.add_constant(xperc)

model_grade = sm.OLS(y, X3).fit()
model_surr = sm.OLS(y, x_surr).fit()
model_perc = sm.OLS(y, xperc).fit()

print(model_grade.summary())
print(model_surr.summary())
print(model_perc.summary())


#multiple regression
xcomb = np.c_(X3 + x_surr + xperc)
xcomb = sm.add_constant(xcomb)
model_mult = sm.OLS(y, xcomb).fit()
print(model_mult.summary())

# Graph 


    

