import numpy as mynp
import pandas as mypd
import matplotlib.pyplot as mympl
import seaborn as mysbn

df = mypd.read_csv('Newyorkairbnbdata.csv')
df_visual = df

df.drop(['host_id','latitude','longitude'], inplace=True, axis=1)

df.isnull().sum()

price_avg = df["price"].mean()
df["price"].fillna(price_avg, inplace = True)

avail_avg = df["availability_365"].mean()
df["availability_365"].fillna(avail_avg, inplace = True)

df[df.duplicated()]

df.drop_duplicates(inplace = True)
