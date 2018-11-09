# Fundamentals of Data Analysis
# An analysis of Anscombe's dataset
# Mary McDonagh

### Table of Contents
- 1.0 Investigation
- 2.0 Assignment Questions:
- 2.1 Explain the background to the dataset – who created it, when it was created, and any speculation you can find regarding how it might have been created.
- 2.2 Plot the interesting aspects of the dataset.
- 2.3 Calculate the descriptive statistics of the variables in the dataset.
- 2.4 Explain why the dataset is interesting, referring to the plots and statistics above
- 3.0 Summary
- 4.0 References

### 1.0 Investigation
Investigate the Anscombe Quartet using Python.

Initial steps carried out:

- Research the backgroud of the dataset.
- Research different plots to show intersting aspects of it.
- Import required libraries. 
- Research a variety of descriptive statistics to use to to run on the dataset.
- Analyse all of the above information to summarise teh dataset in detail.

Some of the popular libraries used for simple random data anslysis, permutations, distributions and seeds include the following:
- numpy
- Matplotlib
- Seaborn
- Pandas

### 2.0 Assignment Questions:
### 2.1 Explain the background to the dataset – who created it, when it was created, and any speculation you can find regarding how it might have been created.

### Background of Anscombe Dataset
In 1973, Francis J. Anscombe constructed the Anscombe's quartet dataset. It is comprised of four datasets with almost identical statistical properties. Saying this they all appear very different when they are represented in a graph. Each dataset consists of eleven (x,y) points. 
Anscombe published a paper in 1973 titled "Graphs in Statistical Analysis". This was published in The American Statistician, Vol. 27, No. 1. (Feb., 1973), pp. 17-21, statistician Francis Anscombe provided the briefiest of abstracts: "Graphs are essential to good statistical analysis". At the time John Tukey had establised the idea of using graphical methods but there was a large amount of skepticism around this. 
Anscombe listed some premises that at the time textbooks were “indoctrinating” people with, like the idea that “numerical calculations are exact, but graphs are rough.” Following on from this he presented a table of numbers. This is where the quartet came about. The table contained four distinct datasets (hence the name Anscombe’s Quartet), each with statistical properties that are identical(to two decimal places), the mean of the x values is 9.0, mean of y values is 7.5, they all have nearly identical variances, correlations, and regression lines (to at least two decimal places).

### 2.2 Plot the interesting aspects of the dataset.

All four plots have the same mean, variance and correlation values.

import pandas as pd  # Import the panda package to use for data manipulation
import matplotlib.pyplot as plt  # Use to create plots
import seaborn as sns  # Powerful plots
from scipy import stats  # Linear regression
import numpy as np  # Quick summary statistics

anscombe = sns.load_dataset("anscombe")

#### Load the anscombe dataset in a dataframe (df). Use sns (seaborn package) to plot the data.
df = sns.load_dataset("anscombe")
df

Input:
#### Display the properties and associated value of each in a table.
data = [['Mean of x in each case:','9 (exact).'], 
        ['Variance of x in each case:','11 (exact).'],
        ['Mean of y in each case:','7.50 (to 2 decimal places).' ],
        ['Variance of y in each case:','4.122 or 4.127 (to 3 decimal places).' ],  
        ['Correlation between x and y in each case:','0.816 (to 3 decimal places).'],
       ['Linear regression line in each case:', 'y = 3.00 + 0.500x (to 2 and 3 decimal places).']]
pd.DataFrame(data, columns=["Property", "Value"])

Output:
![image](https://user-images.githubusercontent.com/36244887/anscombe-dataset/properties.JPG)
<p align="center">
              

    
### 2.3 Calculate the descriptive statistics of the variables in the dataset.


### 2.4 Explain why the dataset is interesting, referring to the plots and statistics above


### 3.0 Summary


### 4.0 References



