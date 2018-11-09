<p align="center">           

<h1><b>Project 2018</b><br></h1>
<h2>Fundamentals of Data Analysis - Mary McDonagh</h2>
<h2>An analysis of Anscombe's dataset</h2>
</br>
</p>

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

### Project Plan
![image](https://user-images.githubusercontent.com/36244887/48293746-7716ee00-e478-11e8-962e-ac31ce0c6037.JPG)

### 2.0 Assignment Questions:
### 2.1 Explain the background to the dataset – who created it, when it was created, and any speculation you can find regarding how it might have been created.

### Background of Anscombe Dataset
In 1973, Francis J. Anscombe constructed the Anscombe's quartet dataset. It is comprised of four datasets with almost identical statistical properties. Saying this they all appear very different when they are represented in a graph. Each dataset consists of eleven (x,y) points. 
Anscombe published a paper in 1973 titled "Graphs in Statistical Analysis". This was published in The American Statistician, Vol. 27, No. 1. (Feb., 1973), pp. 17-21, statistician Francis Anscombe provided the briefiest of abstracts: "Graphs are essential to good statistical analysis". At the time John Tukey had establised the idea of using graphical methods but there was a large amount of skepticism around this. 
Anscombe listed some premises that at the time textbooks were “indoctrinating” people with, like the idea that “numerical calculations are exact, but graphs are rough.” Following on from this he presented a table of numbers. This is where the quartet came about. The table contained four distinct datasets (hence the name Anscombe’s Quartet), each with statistical properties that are identical(to two decimal places), the mean of the x values is 9.0, mean of y values is 7.5, they all have nearly identical variances, correlations, and regression lines (to at least two decimal places).

![image](https://user-images.githubusercontent.com/36244887/48272212-018a2e00-e436-11e8-861c-eefd3d06331b.JPG)

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
![image](https://user-images.githubusercontent.com/36244887/48268947-53c75100-e42e-11e8-98d2-fcfa8dae8e3d.JPG)
    
Input:
anscombe = sns.load_dataset("anscombe")

#Using seaborn package to run a pairplot to show 2 plots displaying x and y as per Anscombe's quartet
#Pairplots will only plot numeric columns in dataset
#Each dot in the scatterplot represents a row in the dataframe
sns.pairplot(df)

![image](https://user-images.githubusercontent.com/36244887/48269773-4ad77f00-e430-11e8-856d-d8be76077ae2.JPG)

### Dataset I
sns.lmplot(x="x",  y="y", data=anscombe.query("dataset == 'I'"), 
          ci=None, scatter_kws={"s": 80});

Output:
![image](https://user-images.githubusercontent.com/36244887/48269856-7a868700-e430-11e8-84a7-8574ce287673.JPG)

The graph above shows a small dataset of two values (x and y) which are pretty closely linearly related. The regression line is y = 3.00 + 0.500x. The correlation between x and y (specifically the Pearson Correlation Coefficient) is .816, pretty close to 1, so you might assume the data points would be close to that line. The x variance is 11 so the average distance away from the mean is about 3.3 while the y variance of 4.12, giving a bit smaller average distance of 2.03.

### Dataset II

#non linear quadratic dataset
sns.lmplot(x="x",  y="y", data=anscombe.query("dataset == 'II'"),
          ci=None, scatter_kws={"s": 80});

Output:
![image](https://user-images.githubusercontent.com/36244887/48270121-2b8d2180-e431-11e8-8df8-b3fa77a9e673.JPG)

### Dataset II
#change the order to 2 and see the difference
sns.lmplot(x="x",  y="y", data=anscombe.query("dataset == 'II'"),
          order =2, ci=None, scatter_kws={"s": 80});

Output:
![image](https://user-images.githubusercontent.com/36244887/48270211-67c08200-e431-11e8-9b26-3e802854830e.JPG)

Figure II shows a relationship between two variables which are clearly not linearly related. They appear to form a parabola closely fit together. Essentially this shape of graph could represent the arc of a tennis ball (x vs y position). 

### Dataset III
#using robust regression
sns.lmplot(x="x",  y="y", data=anscombe.query("dataset == 'III'"),
          ci=None, scatter_kws={"s": 80});

Output:
![image](https://user-images.githubusercontent.com/36244887/48270297-a1918880-e431-11e8-8cfb-c724d0cd5762.JPG)

The data in Figure III is perfectly linear. There is one outlier which causes the fit to be skewed off that perfect line. This outlines the effect a single outlier has on a sample, especially when the sample size is small. You can also compare Figure I with Figure III to see the difference between a close linear correlation (Figure I, e.g. heights vs weights) and a perfect linear correlation (Figure III, e.g. height in inches vs height in centimeters).

#setting robust to true
sns.lmplot(x="x",  y="y", data=anscombe.query("dataset == 'III'"),
          robust=True, ci=None, scatter_kws={"s": 80});
          
![image](https://user-images.githubusercontent.com/36244887/48270385-d6054480-e431-11e8-926a-feb52647e038.JPG)

When we plot these four datasets on an x/y coordinate plane, we can observe that they show the same regression lines but each dataset is telling a different story. Dataset I appears to have clean and well-fitting linear models. Dataset II is not distributed normally. In Dataset III the distribution is linear but the calculated regression is thrown off by an outlier.

### Plotting the dataset with Seaborne's "Implot" linear model plot. The line is drawn at best fit through the points given. Seaborn's lmplot uses a combination of regplot() and FacetGrid.

%matplotlib inline
#Show the results of a linear regression within each dataset
sns.lmplot(x="x", y="y", col="dataset", hue="dataset", data=df,
           col_wrap=2, ci=None, palette="husl")
           
![image](https://user-images.githubusercontent.com/36244887/48270471-1664c280-e432-11e8-93f4-fa1f5c4fe967.JPG)

Input:
#show 4 datasets of x and y that have the same mean, standard deviation, and regression line, but which are qualitatively different.

import matplotlib.pyplot as plt
import numpy as py

import matplotlib.pyplot as plt
import numpy as np

x = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5])
y1 = np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68])
y2 = np.array([9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74])
y3 = np.array([7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73])
x4 = np.array([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8])
y4 = np.array([6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89])


def fit(x):
    return 3 + 0.5 * x

xfit = np.array([np.min(x), np.max(x)])

plt.subplot(221)
plt.plot(x, y1, 'ks', xfit, fit(xfit), 'r-', lw=2)
plt.axis([2, 20, 2, 14])
plt.setp(plt.gca(), xticklabels=[], yticks=(4, 8, 12), xticks=(0, 10, 20))
plt.text(3, 12, 'I', fontsize=20)

plt.subplot(222)
plt.plot(x, y2, 'ks', xfit, fit(xfit), 'r-', lw=2)
plt.axis([2, 20, 2, 14])
plt.setp(plt.gca(), xticks=(0, 10, 20), xticklabels=[],
         yticks=(4, 8, 12), yticklabels=[], )
plt.text(3, 12, 'II', fontsize=20)

plt.subplot(223)
plt.plot(x, y3, 'ks', xfit, fit(xfit), 'r-', lw=2)
plt.axis([2, 20, 2, 14])
plt.text(3, 12, 'III', fontsize=20)
plt.setp(plt.gca(), yticks=(4, 8, 12), xticks=(0, 10, 20))

plt.subplot(224)
xfit = np.array([np.min(x4), np.max(x4)])
plt.plot(x4, y4, 'ks', xfit, fit(xfit), 'r-', lw=2)
plt.axis([2, 20, 2, 14])
plt.setp(plt.gca(), yticklabels=[], yticks=(4, 8, 12), xticks=(0, 10, 20))
plt.text(3, 12, 'IV', fontsize=20)

#verify the stats
pairs = (x, y1), (x, y2), (x, y3), (x4, y4)
for x, y in pairs:
    print('mean=%1.2f, std=%1.2f, r=%1.2f' % (np.mean(y), np.std(y),
          np.corrcoef(x, y)[0][1]))

plt.show()

Output:
![image](https://user-images.githubusercontent.com/36244887/48270611-588e0400-e432-11e8-82fb-f50db5bd9d84.JPG)

#### Common to these datasets:
Mean of x: 9 Sample variance of x: 11 Mean of y: 7.50 Sample variance of y: 4.12 Linear regression: y=3.00+0.500*x R squared: 0.666 p value 0.0021

Figure IV is a more evident display of the effect of an outlier on a sample. From the graph, it is evident that a linear fit between x and y doesn’t make sense. Seeing the data in this shape and graph may make you think that it has come from two different samples. While summary statistics are not enough to describe a dataset on their own, in conjunction with a graph they can be quite useful. Figure 1 shows a clear linear correlation, graphically, and the sumary statistics help describe it even better and more precisely. Using Plotly we can generate these statistics easily. Ideally comprehensive data analysis consists of both numerical statistics and clear visualisations. 
In this 1973 paper, Anscombe summises with a call to action: “The user is not showered with graphical displays. He can get them only with trouble, cunning and a fighting spirit. It’s time that was changed.” Plotly has allowed for these changes to take place. 

### 2.3 Calculate the descriptive statistics of the variables in the dataset.

import pandas as pd  # Import the panda package to use for data manipulation
import matplotlib.pyplot as plt  # Use to create plots
import seaborn as sns  # Powerful plots
from scipy import stats  # Linear regression
import numpy as np  # Quick summary statistics

anscombe = sns.load_dataset("anscombe")

df = sns.load_dataset("anscombe")
df

Input:
#Describe the dataframe outlining count, mean, standard deviation, min and max. Include headings in the data.
overview = df.describe()
overview = overview.transpose()
overview.head()

Output:
![image](https://user-images.githubusercontent.com/36244887/48270812-d6520f80-e432-11e8-95be-0a812d1e07bd.JPG)

Input:
#Desribe and break down an overview of the four datasets per row with comprehensive count, mean, standard deviation, min and max vales.
anscombe.groupby("dataset").describe()

Output:
![image](https://user-images.githubusercontent.com/36244887/48270926-11544300-e433-11e8-8967-5d11c4cec968.JPG)

Input:
#Use .shape to define the # of rows and columns in the dataframe (df).
df.shape

Output:
(44, 3)

Input:
#List in y.
df.loc[:,'y']


Input:
#result of y.
df.at[5,'y']

Output:
9.96

Input:
#Display the first five lines of data to ensure it has imported without any issues.
df = sns.load_dataset("anscombe")
df.head()

Output:
![image](https://user-images.githubusercontent.com/36244887/48271096-74de7080-e433-11e8-9c21-712bbe8e75f7.JPG)

Input:
#Display the last 5 rows of data.
df.tail()

Output:
![image](https://user-images.githubusercontent.com/36244887/48271167-a3f4e200-e433-11e8-900a-a31ee048be80.JPG)

Input:
#Describe the dataframe including count (# of rows), mean, standard deviation(measure of the spread of the values), min and max values.
#Percentages outline what % of the figures are less than 25/50/75. 50% is often referred to as the medium. E.g. 25% of x values are less than 7
df.describe()

Output:
![image](https://user-images.githubusercontent.com/36244887/48271240-d30b5380-e433-11e8-96ca-48e820ed5b71.JPG)

Input:
#Display data in rows 5-7.
df.loc[5:7]

Output:
![image](https://user-images.githubusercontent.com/36244887/48271328-ffbf6b00-e433-11e8-920f-73ea0d4eae0f.JPG)

Input
#Define the mean values of x and y
df.mean()

Output:
x    9.000000
y    7.500682
dtype: float64

Input:
#Use the describe function to display an overview of the data e.g. mean, count, min and max value, standard deviation.
df.groupby("dataset").describe()

Output:
![image](https://user-images.githubusercontent.com/36244887/48271451-4319d980-e434-11e8-9c34-ff540ee08989.JPG)

Input:
#Display the Anscombe's quartet (four datasets) in a table

anscombes_data = {'X1':[10,8,13,9,11,14,6,4,12,7,5],
            'X2': [10,8,13,9,11,14,6,4,12,7,5],
            'X3': [10,8,13,9,11,14,6,4,12,7,5],
            'X4': [8,8,8,8,8,8,8,19,8,8,8],
            'Y1': [8.04,6.95,7.58,8.81,8.33,9.96,7.24,4.26,10.84,4.82,5.68],
            'Y2': [9.14,8.14,8.74,8.77,9.26,8.1,6.13,3.1,9.13,7.26,4.74],
            'Y3': [7.46,6.77,12.74,7.11,7.81,8.84,6.08,5.39,8.15,6.42,5.73],
            'Y4': [6.58,5.76,7.71,8.84,8.47,7.04,5.25,12.5,5.56,7.91,6.89]}

df = pd.DataFrame(anscombes_data)
df

Output:
![image](https://user-images.githubusercontent.com/36244887/48271544-765c6880-e434-11e8-9991-c9e3f56cc282.JPG)

Input:
# Display the mean values
print(df.mean())

Output:
X1    9.000000
X2    9.000000
X3    9.000000
X4    9.000000
Y1    7.500909
Y2    7.500909
Y3    7.500000
Y4    7.500909
dtype: float64

Input:
#Display the variances
print(df.var())

Output:
X1    11.000000
X2    11.000000
X3    11.000000
X4    11.000000
Y1     4.127269
Y2     4.127629
Y3     4.122620
Y4     4.123249
dtype: float64

Input:
#Display the correlation of the dataframe
df['X1'].corr(df['Y1']),df['X2'].corr(df['Y2']),df['X3'].corr(df['Y3']),df['X4'].corr(df['Y4'])

Output:
(0.81642051634484, 0.8162365060002427, 0.8162867394895982, 0.8165214368885028)

### 2.4 Explain why the dataset is interesting, referring to the plots and statistics above

Anscombe's Quartet highlights the importance of plotting your data and not relying on statistics alone to anyslyse data. This quartet is four separate and distinct datasets with very different outputs once graphed. When we graph the data and apply the linear regression we can identify that while all four have the same summary statistics they do not plot with the same output. The summary statistics displays the same mean or x and y, same variance or x and y, same correlation, same slope of x and y, same intercept of the line. The lesson to be learned from the Anscombe Quartet is that summary statistics do not tell the whole story. 
Anscombe’s quartet is regulary used as an example to outline that a summary of a dataset will lose information and should always be followed with additional study or plotting of data. There is a paper which shows how to generate similar data sets (behind a paywall): “Generating Data with Identical Statistics but Dissimilar Graphics: A Follow up to the Anscombe Dataset”). 
Anscombe concluded his paper as follows:
"Graphical output such as described above is readily available to anyone who does his own programming. I myself habitually generate such plots at an APL terminal, and have come to appreciate their importance. A skilled Fortran or PL/1 programmer, with an organized library of subroutines, can do the same (on a larger scale). Unfortunately, most persons who have recourse to a computer for statistical analysis of data are not much interested either in computer programming or in statistical method, being primarily concerned with their own proper business. Hence the common use of library programs and various statistical packages. Most of these originated in the pre-visual era. The user is not showered with graphical displays. He can get them only with trouble, cunning and a fighting spirit. It's time that was changed."
Anscombe’s Quartet are certainly not the only set of numbers that display a discrepancy between the results of statistical summary and results of a plotted graph. In 2007 Chatterjee and Firat published a paper which outlined a method for generating datasets that looked the same statistically but when plotted displayed different results. Additionally in Other similar work 2009, Haslett & Govindaraju provided information for generating datasets with similar multiple linear regression. These papers confirm that Anscombe’s Quartet is not just an example but there are an infinite amount of sets that would look almost identical in traditional statistical analysis and display a different graphical output.

### 3.0 Summary
Throughout this project I have reviewed the background of the dataset and provide an overview of the history, creator and premise behind it. I have used the research to allow me to plot numerous graphs displaying the dataset and what makes it interesting. Following on from the graphs I have outlined and calculated the descriptive statistics in relation to the dataset which allows us to analyse it further. Finally I provided an explanation of the dataset based on my research in the project. To summarise, Anscombe's Quartet contained four distinct datasets, each with identical statistical properties (to two decimal places), the mean of the x values is 9.0, mean of y values is 7.5, they all have nearly identical variances, correlations, and regression lines (to at least two decimal places).

### 4.0 References
#http://complementarytraining.net/stats-playbook-what-is-anscombes-quartet-and-why-is-it-important/
#https://www.jstor.org/stable/2682899?seq=1#page_scan_tab_contents
#'https://matplotlib.org/gallery/specialty_plots/anscombe.html'
#https://heapanalytics.com/blog/data-stories/anscombes-quartet-and-why-summary-statistics-dont-tell-the-whole-story
#https://eagereyes.org/criticism/anscombes-quartet
#https://plotlyblog.tumblr.com/post/68951620673/why-graph-anscombes-quartet
#https://vknight.org/unpeudemath/mathematics/2016/10/29/anscombes-quartet-variability-and-ciw.html
#http://nbviewer.jupyter.org/github/psychemedia/ou-tm351/blob/master/notebooks-RFC/Anscombe's%20Quartet%20%5Bopen%5D.ipynb
#https://www.asis.org/asist2013/proceedings/submissions/posters/41poster.pdf


