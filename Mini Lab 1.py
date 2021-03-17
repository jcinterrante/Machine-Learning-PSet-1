# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sn

os.chdir("C:/Users/jcinterrante/OneDrive - The University of Chicago/Classes/Machine Learning/HW/Lab HW 1")

acs_data = pd.read_csv("usa_00002.csv")
crosswalk = pd.read_csv("educ crosswalk.csv")

# Use the educd variable to create a continuous measure of education called educdc using the crosswalk
acs_data['EDUCD'].unique()

acs_data = acs_data.merge(crosswalk, left_on="EDUCD", right_on="educd")

# Create dummy variables hsdip, coldip, white, black, hispanic, married, female, vet
acs_data["hsdip"] = np.where(np.isin(acs_data["EDUCD"], range(62, 84))|np.isin(acs_data["EDUCD"], range(90, 101)), 1, 0)
acs_data["coldip"] = np.where(np.isin(acs_data["EDUCD"], range(101, 117)), 1, 0)
acs_data["white"] = np.where(acs_data["RACE"] == 1, 1, 0)
acs_data["black"] = np.where(acs_data["RACE"] == 2, 1, 0)
acs_data["hispanic"] = np.where(np.isin(acs_data["HISPAN"], 1, 4), 1, 0)
acs_data["married"] = np.where(np.isin(acs_data["MARST"], 1|2), 1, 0) 
acs_data["female"] = np.where(acs_data["SEX"] == 2, 1, 0) 
acs_data["vet"] = np.where(acs_data["VETSTAT"] == 2, 1, 0)

# Create an interaction between each of the education dummy variables (A-B) and education.
acs_data["educ_x_hs"] = acs_data["educdc"] * acs_data["hsdip"]
acs_data["educ_x_col"] = acs_data["educdc"] * acs_data["coldip"]

# Create the following: Age squared. The natural log of incwage.
acs_data["age_squared"] = np.power(acs_data["AGE"], 2)
acs_data = acs_data[acs_data['INCWAGE'] > 0]
acs_data["incwage_log"] = np.log(acs_data["INCWAGE"])



# Compute descriptive (summary) statistics for the following variables: year, 
# incwage, lnincwage, educdc, female, age, age2, white, black, hispanic, 
# married, nchild, vet, hsdip, coldip, and the interaction terms. In other 
# words, compute sample means, standard deviations, etc.
summary_cols = ["YEAR", "EDUC", "educdc", "female", "AGE", "age_squared",
                "white", "black", "hispanic", "married", "NCHILD","vet","hsdip",
                "coldip", "educ_x_hs", "educ_x_col"]
summary = acs_data.describe(include="all")[summary_cols]

# Scatter plot ln(incwage) and education. Include a linear fit line. Be sure to 
# label all axes and include an informative title.

plt.scatter(acs_data["educdc"], acs_data["incwage_log"], alpha = 0.1)
sn.regplot(x="educdc", y="incwage_log", data = acs_data)
plt.title("Wages tend to rise with greater education")
plt.xlabel("Education")
plt.ylabel("Income (log dollars)")


# Estimate a linear model of ln incwage and report your results
income_log_lm = smf.ols("incwage_log ~ educdc + female + AGE + age_squared + white + black + hispanic + married + NCHILD + vet", data = acs_data).fit()

print(income_log_lm.summary())

# Test the hypothesis that race has no effect on wages. Be sure to explicitly state the null
# and alternative hypotheses and show your calculations.
income_race_log_lm = smf.ols("incwage_log ~ white + black + hispanic", data = acs_data).fit()
print(income_race_log_lm.summary())


RSS = income_log_lm.ssr
RSS_0 = income_race_log_lm.ssr

print("F = ",((RSS_0-RSS)/7)/(RSS/(14900)))

print(income_log_lm.compare_f_test(income_race_log_lm)[1])
print("COMPARISON\n",income_log_lm.compare_f_test(income_race_log_lm))

# 4. Graph ln(incwage) and education. Include three distinct linear fit lines 
# specific to individuals with no high school diploma, a high school diploma, 
# and a college degree. Be sure to label all axis and include an informative 
# title.
no_highschool = acs_data.query("hsdip == 0")
highschool = acs_data.query("hsdip == 1")
college = acs_data.query("coldip == 1")

sn.regplot(x="educdc", y="incwage_log", data = no_highschool)
plt.title("Wages and Education - No Higschool Diploma")
plt.xlabel("Education")
plt.ylabel("Income (log dollars)")

sn.regplot(x="educdc", y="incwage_log", data = highschool)
plt.title("Wages and Education - Higschool Diploma")
plt.xlabel("Education")
plt.ylabel("Income (log dollars)")

sn.regplot(x="educdc", y="incwage_log", data = college)
plt.title("Wages and Education - College Diploma")



# 6. Estimate the model you proposed in the previous question and report your results.
income_log_lm_expanded = smf.ols("incwage_log ~ educdc + female + AGE + age_squared + white + black + hispanic + married + NCHILD + vet + hsdip + coldip + educdc* hsdip + educdc*coldip", data = acs_data).fit()
print(income_log_lm_expanded.summary())

# (a) Predict the wages of an 22 year old, female individual (who is neither white, black,
# nor Hispanic, is not married, has no children, and is not a veteran) with a high school
# diploma and an all else equal individual with a college diploma. Assume that it takes
# someone 12 years to graduate high school and 16 years to graduate college.
person_no_college_dict = {"female":[1],
          "AGE":[22],
          "age_squared":[484],
          "white":[0],
          "black":[0],
          "hispanic":[0],
          "married":[0],
          "NCHILD":[0],
          "vet":[0],
          "hsdip":[1],
          "coldip":[0],
          "educdc":[12]}
person_no_college = pd.DataFrame(data=person_no_college_dict)

person_college_dict = {"female":[1],
          "AGE":[22],
          "age_squared":[484],
          "white":[0],
          "black":[0],
          "hispanic":[0],
          "married":[0],
          "NCHILD":[0],
          "vet":[0],
          "hsdip":[1],
          "coldip":[1],
          "educdc":[16]}
person_college = pd.DataFrame(data=person_college_dict)

prediction_no_college = income_log_lm_expanded.get_prediction(person_no_college)
print(prediction_no_college.summary_frame(alpha=0.05))

prediction_college = income_log_lm_expanded.get_prediction(person_college)
print(prediction_college.summary_frame(alpha=0.05))
#5
#(b) The President wants to know, given your results, do individuals with college degrees
#have higher predicted wages than those without? By how much? Briefly explain.
#(c) The President asked you to look into this question because she is considering legislation
#that will expand access to college education (for instance, by increasing student loan
#subsidies). She will only support the legislation if there are cost offsets (if college
#education increases wages and therefore, future income tax revenues that help reduce
#the net cost of the subsidy). Given that criteria, how would you advise the President?
#7. There are many ways that this model could be improved. How would you do things differently if you were asked to predict the returns to education given the data available on
#IPUMS?
