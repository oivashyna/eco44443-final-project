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
from statsmodels.multivariate.manova import MANOVA

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
# --- HOLC grade (numeric) ---
X3 = redline_full[["holc_grade_x"]].copy()
X3["holc_grade_x"] = X3["holc_grade_x"].map({"A": 4, "B": 3, "C": 2, "D": 1})
X3 = sm.add_constant(X3)

# --- surrounding percentages ---
x_surr = redline_full[
    ["surr_area_pct_white","surr_area_pct_black","surr_area_pct_hisp",
     "surr_area_pct_asian","surr_area_pct_other"]
].copy()
x_surr = sm.add_constant(x_surr)

# --- G metrics ---
xperc = redline_full[
    ["avg_G","median_G","decile_10_G","quartile_25_G"]
].copy()
xperc = sm.add_constant(xperc)

y_cols = ["pct_white","pct_black","pct_hisp","pct_asian","pct_other"]
Y = redline_full[y_cols]
for col in y_cols:
    y = redline_full[col]
    
    print(f"\n===== Dependent variable: {col} =====")
    
    model_grade = sm.OLS(y, X3).fit()
    print("\n-- Grade model --")
    print(model_grade.summary())
    
    model_surr = sm.OLS(y, x_surr).fit()
    print("\n-- Surrounding model --")
    print(model_surr.summary())
    
    model_perc = sm.OLS(y, xperc).fit()
    print("\n-- G metrics model --")
    print(model_perc.summary())

#multiple regression - check over this

formula = "pct_white + pct_black + pct_hisp + pct_asian + pct_other ~ holc_grade_x + surr_area_pct_white + surr_area_pct_black + surr_area_pct_hisp + surr_area_pct_asian + surr_area_pct_other + avg_G + median_G + decile_10_G + quartile_25_G"

model = MANOVA.from_formula(formula, data=redline_full)
print(model.mv_test())

# Graph 
plt.style.use('classic')
# race vs HOLC grade
pct_cols = ["pct_white", "pct_black", "pct_hisp", "pct_asian", "pct_other"]
g_cols = ["avg_G", "median_G", "decile_10_G", "quartile_25_G"]
surr_cols = [
    "surr_area_pct_white",
    "surr_area_pct_black",
    "surr_area_pct_hisp",
    "surr_area_pct_asian",
    "surr_area_pct_other"
]


for grade in ["A", "B", "C", "D"]:
    indicator = (redline_full["holc_grade_x"] == grade).astype(int)
    
    # jitter to avoid vertical stacking
    x = indicator + np.random.normal(0, 0.02, size=len(indicator))
    
    for col in pct_cols + g_cols:
        plt.figure()
        plt.scatter(x, redline_full[col])
        plt.xlabel(f"{grade} indicator (jittered)")
        plt.ylabel(col)
        plt.title(f"{col} vs HOLC Grade {grade}")
        plt.show()

# race vs surrounding area
for surr in surr_cols:
    for pct in pct_cols:
        plt.figure()
        plt.scatter(redline_full[surr], redline_full[pct])
        plt.xlabel(surr)
        plt.ylabel(pct)
        plt.title(f"{pct} vs {surr}")
        plt.show()    

# race vs 2020 percentile
g_cols = ["avg_G", "median_G", "decile_10_G", "quartile_25_G"]
pct_cols = ["pct_white", "pct_black", "pct_hisp", "pct_asian", "pct_other"]

for g in g_cols:
    for pct in pct_cols:
        plt.figure()
        plt.scatter(redline_full[pct], redline_full[g])
        plt.xlabel(pct)
        plt.ylabel(g)
        plt.title(f"{g} vs {pct}")
        plt.show()

# G metrics vs surrounding %

for g in g_cols:
    for surr in surr_cols:
        plt.figure()
        plt.scatter(redline_full[surr], redline_full[g])
        plt.xlabel(surr)
        plt.ylabel(g)
        plt.title(f"{g} vs {surr}")
        plt.show()

for g in g_cols:
    for surr in surr_cols:
        for pct in pct_cols:
            plt.figure()
            plt.scatter(redline_full[surr], redline_full[g], redline_full[pct])
            plt.xlabel(surr, g)
            plt.ylabel(pct)
            plt.title(f"{surr, g} vs {pct}")
            plt.show()