# -*- coding: utf-8 -*-
"""New York Airbnb.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HnPcC-nv1mBrPEqx8hpa9zb05e3-nDD9

# **NEWYORK AIRBNB**

# **About Dataset:**

**New York Airbnb Data**

Airbnb is an online marketplace for short-term homestays and experiences. Airbnb acts as a broker and charges a commission from each booking. This project is to analyze New York Airbnb data.

The data have 12 columns and 48,896 rows. The column is as follows:
- host_id	
- neighbourhood_group	
- neighbourhood	
- latitude	
- longitude	
- room_type	
- price	
- minimum_nights	
- number_of_reviews	
- reviews_per_month	
- calculated_host_listings_count	
- availability_365

There is no dependent variables in New York Airbnb Data and therefore I will use unsupervised learning for the project.
I use K-Means Clustering algorithm to evaluate the Airbnb data.

# **Step 1:Import necessary libraries**
"""

import numpy as mynp

import pandas as mypd

import matplotlib.pyplot as mympl

import seaborn as mysbn

import math

"""# **Step 2:Import the data inside google colab**"""

df = mypd.read_csv('/content/Newyorkairbnbdata.csv')

df_visual = df

"""# **Step 3:Data Exploration**"""

df.head()

df.dtypes

df.info()

df.shape

df.describe()

"""# **Step 4:Data Cleaning**

Check for missing values,duplicate values,categorical values and outliers and handle them accordingly.

Consider the table is read using variable "df" Then, use the functions like


1.   Check for the column datas types and customise the datatype if necessary
2.   df.isna().sum()
3.   df['colname'].fillna()
4.   df[df.duplicated()]
5.   df.drop_duplicates()
6.   Use box plot to check for outliers
7.   Remove the outliers by any technique
8.   Scaling of numerical features
9.   Encode the categorical data into numerical
10.  Display the clean data


"""

#Remove unnecessary column
df.drop(['host_id','latitude','longitude'], inplace=True, axis=1)

df.head()

df.isnull().sum()

#Replace '0' in column since it does not make sense for price to be zero. mode method. change to mean
price_avg = df["price"].mean()
print(price_avg)
df["price"].fillna(price_avg, inplace = True)

#Replace '0' in column since it does not make sense for room availibility to be zero. mean method.
avail_avg = df["availability_365"].mean()
print(avail_avg)
df["availability_365"].fillna(avail_avg, inplace = True)

df[df.duplicated()]

#Method to remove duplicates. My data have no duplicate
df.drop_duplicates(inplace = True)

df.boxplot()

mysbn.boxplot(y=df["price"])

df['price'].plot(kind="box")
mympl.show()

df['number_of_reviews'].plot(kind="box")

df['reviews_per_month'].plot(kind="box")

df['availability_365'].plot(kind="box")

#import scipy.stats as stats
#df['ZR'] = stats.zscore(df['number_of_reviews'])
#df['number_of_reviews'].mean()
#df['number_of_reviews'].std()

#df[(df['ZR']<-3) | (df['ZR']>3)]
#df[(df['ZR']<-3) | (df['ZR']>3)].shape[0]

#df2= df[(df['ZR']>-3) & (df['ZR']<3)].reset_index()
#df2.head()

#df.shape

#df2.shape

#df3 = df2.copy()
#df3.drop(columns = ['ZR'], inplace=True)
#df3.head()

df['neighbourhood_group'].unique()

df['neighbourhood'].unique()

df['room_type'].unique()

from sklearn.preprocessing import LabelEncoder
  
myLE = LabelEncoder()
# Encode labels in column that not numerical.
df['neighbourhood_group'] = myLE.fit_transform(df['neighbourhood_group']) 
df['neighbourhood'] = myLE.fit_transform(df['neighbourhood']) 
df['room_type'] = myLE.fit_transform(df['room_type'])

df['neighbourhood_group'].unique()

df['room_type'].unique()

df['neighbourhood'].unique()

df.dtypes

df.head()

"""# **Step 5:Data Visualization**
Explain the findings on visualizing the data
(Create atleast 5 charts using matplotlib and seaborne)and one chart using sweet Viz tool

"""

df_visual['neighbourhood_group'].value_counts()

#df['neighbourhood_group'] = myLE.inverse_transform(df.visual['neighbourhood_group'])

from pandas.core.groupby import grouper
group = ['Brooklyn', 'Manhattan', 'Queens', 'Staten Island', 'Bronx']
df_visual["neighbourhood_group"].plot(kind = 'hist') #is in numeric value
mympl.title("Number of available Airbnb in each Neighbourhood")
mympl.xlabel('Neighbourhood Group') 
mympl.ylabel('Number of Airbnb')
mympl.grid()

"""The graph shows the number of available Airbnb in each Neighbourhood group in New York. Manhattan has the highest number of Airbnb while Bronx has the lowest.

0 - Staten Island
1 - Manhattan
2 - Brooklyn
3 - Queens
4 - Bronx
"""

a = df_visual['room_type'].value_counts()

room = ['Private room','Entire home/apt','Shared room']
cols = ['c', 'r', 'm']
mympl.figure(figsize=(5,5))
mympl.pie(a,labels=room,colors=cols,explode=(0.0,0.0,0.0),autopct='%1.2f%%')
mympl.title('Room Type in Airbnb')
mympl.show()

"""This pie chart is to show percentage of room type listed in Airbnb. We can see that number of private room listed in Airbnb is the highest"""

x = df_visual['neighbourhood']
y = df_visual['price']

mympl.scatter(x, y, label='Ney York neighbourhood', color='c')
mympl.xlabel('Neighbourhood')
mympl.ylabel('Price(USD)')
mympl.title('Scatter Plot')
mympl.legend()
mympl.show()

"""This scatter plot is to see the trend of the Airbnb price in New York. Average Airbnb price is below USD 1000. The highest price is at USD 10,000."""

mysbn.barplot(data = df_visual,x = 'neighbourhood_group', y = 'number_of_reviews')

"""The bar plot is to show number of review received for each neighbourhood group.

* 0 - Brooklyn
* 1 - Manhattan
* 2 - Queens
* 3 - Staten Island
* 4 - Bronx


Staten Island has the number of review out of all neighbourhood




"""

mysbn.violinplot(x = df_visual['number_of_reviews'], palette = 'rainbow');

!pip install sweetviz
import sweetviz as mysv
Airbnb_report = mysv.analyze(df_visual)
Airbnb_report.show_html('Newyorkairbnbdata.html')
Airbnb_report.show_notebook()

"""# **Step 6: Data Splitting into train and test**"""

#Unsupervised learning does not required data splitting

"""# **Step 7: Model Building**


"""

df.head()

from sklearn.cluster import KMeans

sse = []
for k in range(1, 15):
   km = KMeans(n_clusters=k)
   km.fit(df)
   sse.append(km.inertia_)

mympl.figure(figsize=(6, 6))
mympl.plot(range(1, 15), sse, '-o', c = 'maroon')
mympl.xlabel('Count of Clusters')
mympl.ylabel('SSE');

"""Airbnb data have 5 clusters"""

kmeans_model = KMeans(n_clusters=5)
y_kmeans = kmeans_model.fit_predict(df)

y_kmeans

centers = kmeans_model.cluster_centers_
mympl.scatter(centers[:, 0], centers[:, 1], c='black', s=300, alpha=0.2);

#kmeans.plot_kmeans()
from sklearn.cluster import KMeans, SpectralClustering

#plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
from KMeansAlgorithm import KMeansAlgorithm

kmeans = KMeansAlgorithm(df, 5) # setting K=5
kmeans.fit_model(100) # 100 iterations
kmeans.plot_kmeans()

#mympl.scatter(df,y_kmeans)

#filter rows of original data
filtered_label2 = df[y_kmeans == 2]
 
filtered_label8 = df[y_kmeans == 5]
 
#Plotting the results
mympl.scatter(filtered_label2[:,0] , filtered_label2[:,1] , color = 'red')
mympl.scatter(filtered_label8[:,0] , filtered_label8[:,1] , color = 'black')
mympl.show()

filtered_label2



#import factoextra
#library("factoextra")

fviz_cluster(kmeans, data = df,
             palette = c("red", "blue", "cyan","yellow","green"), 
             geom = "point",
             ggtheme = theme_bw()
             )

"""# **Step 8: Model Validation**"""

mympl.scatter(df[], c=y_kmeans, cmap='viridis')

centers = kmeans.cluster_centers_
mympl.scatter(centers[:, 0], centers[:, 1], c='black', s=300, alpha=0.2);

"""# **Step 9:Model Evaluation and Visualization**"""

from sklearn.metrics import silhouette_score  
ss = silhouette_score(df, kmeans.labels_)
print(ss)

"""#**Step 10: Creating WebApp using Streamlit**

Refer to Nash Google Site - https://sites.google.com/view/nashswebsite/machine-learning-project
"""