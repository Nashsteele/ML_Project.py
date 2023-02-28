import numpy as mynp
import pandas as mypd
import matplotlib.pyplot as mympl
import seaborn as mysbn
import math

df = mypd.read_csv('/content/Newyorkairbnbdata.csv')
df_visual = df

df.drop(['host_id','latitude','longitude'], inplace=True, axis=1)

df.isnull().sum()

price_avg = df["price"].mean()
df["price"].fillna(price_avg, inplace = True)


avail_avg = df["availability_365"].mean()
df["availability_365"].fillna(avail_avg, inplace = True)

df[df.duplicated()]

df.drop_duplicates(inplace = True)

df.boxplot()

mysbn.boxplot(y=df["price"])

df['price'].plot(kind="box")
mympl.show()

df['number_of_reviews'].plot(kind="box")

df['reviews_per_month'].plot(kind="box")

df['availability_365'].plot(kind="box")


from sklearn.preprocessing import LabelEncoder
  
myLE = LabelEncoder()
df['neighbourhood_group'] = myLE.fit_transform(df['neighbourhood_group']) 
df['neighbourhood'] = myLE.fit_transform(df['neighbourhood']) 
df['room_type'] = myLE.fit_transform(df['room_type'])

from pandas.core.groupby import grouper
group = ['Brooklyn', 'Manhattan', 'Queens', 'Staten Island', 'Bronx']
df_visual["neighbourhood_group"].plot(kind = 'hist') #is in numeric value
mympl.title("Number of available Airbnb in each Neighbourhood")
mympl.xlabel('Neighbourhood Group') 
mympl.ylabel('Number of Airbnb')
mympl.grid()

a = df_visual['room_type'].value_counts()
room = ['Private room','Entire home/apt','Shared room']
cols = ['c', 'r', 'm']
mympl.figure(figsize=(5,5))
mympl.pie(a,labels=room,colors=cols,explode=(0.0,0.0,0.0),autopct='%1.2f%%')
mympl.title('Room Type in Airbnb')
mympl.show()

x = df_visual['neighbourhood']
y = df_visual['price']

mympl.scatter(x, y, label='Ney York neighbourhood', color='c')
mympl.xlabel('Neighbourhood')
mympl.ylabel('Price(USD)')
mympl.title('Scatter Plot')
mympl.legend()
mympl.show()

mysbn.barplot(data = df_visual,x = 'neighbourhood_group', y = 'number_of_reviews')

mysbn.violinplot(x = df_visual['number_of_reviews'], palette = 'rainbow');

"""!pip install sweetviz
import sweetviz as mysv
Airbnb_report = mysv.analyze(df_visual)
Airbnb_report.show_html('Newyorkairbnbdata.html')
Airbnb_report.show_notebook()"""


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


kmeans_model = KMeans(n_clusters=5)
y_kmeans = kmeans_model.fit_predict(df)

centers = kmeans_model.cluster_centers_
mympl.scatter(centers[:, 0], centers[:, 1], c='black', s=300, alpha=0.2);

"""
from sklearn.cluster import KMeans, SpectralClustering

from KMeansAlgorithm import KMeansAlgorithm

kmeans = KMeansAlgorithm(df, 5) # setting K=5
kmeans.fit_model(100) # 100 iterations
kmeans.plot_kmeans()


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

"""mympl.scatter(df[], c=y_kmeans, cmap='viridis')

centers = kmeans.cluster_centers_
mympl.scatter(centers[:, 0], centers[:, 1], c='black', s=300, alpha=0.2);

"""# **Step 9:Model Evaluation and Visualization**"""

from sklearn.metrics import silhouette_score  
ss = silhouette_score(df, kmeans.labels_)
print(ss)
