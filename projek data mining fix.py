#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


# In[2]:


class ColumnsData:
    date = 'Date'
    province = 'Province'
    island = 'Island'
    cases = 'Total Cases'
    deaths = 'Total Deaths'
    recovered = 'Total Recovered'
    actives_cases = 'Total Active Cases'
    population = 'Population'
    area = 'Area (km2)'
    mortality = 'Mortality'
    density = 'Population Density'


# In[3]:


def create_bin(df, columns, q=5):
    for column in columns:
        df[column] = pd.qcut(df[column], q, duplicates='drop').cat.codes


# In[4]:


def normalisasi_data(df, columns):
    minMaxScaler = MinMaxScaler()
    df[columns] = minMaxScaler.fit_transform(d[columns])


# In[5]:


data = pd.read_csv('covid19.csv')
pd.options.display.max_columns = None
data.head().T


# In[6]:


data.shape


# In[7]:


data.isnull().sum()


# In[8]:


data.dtypes


# In[9]:


data = data[[
    ColumnsData.date,
    ColumnsData.province,
    ColumnsData.island,
    ColumnsData.cases,
    ColumnsData.deaths,
    ColumnsData.recovered,
    ColumnsData.actives_cases,
    ColumnsData.population,
    ColumnsData.area,
    ColumnsData.density
]]


# In[10]:


data.isnull().sum()


# In[11]:


data = data.dropna(axis=0, how="any")


# In[12]:


data.isnull().sum()


# In[13]:


data.head()


# In[14]:


data['Total Active Cases'] = data['Total Active Cases'].clip(lower=0)


# In[15]:


data[ColumnsData.date] = pd.to_datetime(data.Date, infer_datetime_format=True).dt.date
data.head()


# In[16]:


data[ColumnsData.mortality] = data[ColumnsData.deaths] / data[ColumnsData.cases]


# In[17]:


data.head().T


# In[19]:


dfl = data[[ColumnsData.date, ColumnsData.cases, ColumnsData.deaths, 
            ColumnsData.recovered]].groupby(ColumnsData.date).sum().reset_index()

dfl = dfl[(dfl[ColumnsData.cases] >= 100)].melt(id_vars=ColumnsData.date,
                                    value_vars=[ColumnsData.cases, 
                                    ColumnsData.deaths, ColumnsData.recovered])


# In[20]:


plot_a = px.line(dfl, x=ColumnsData.date, y='value', color='variable', template="plotly_white")
plot_a.update_layout(title='COVID-19 in Indonesia: total number of cases over time',
                     xaxis_title='Indonesia', yaxis_title='Number of cases',
                     legend=dict(x=0.02, y=0.98))
plot_a.show()


# In[21]:


pd.options.mode.chained_assignment = None
limit = 5
group = data.groupby(ColumnsData.province)
t = group.tail(1).sort_values(ColumnsData.cases, ascending=False).set_index(ColumnsData.province).drop(
    columns=[ColumnsData.date])

s = data[(data[ColumnsData.province].isin([i for i in t.index[:limit]]))]
s = s[(s[ColumnsData.cases] >= 1000)]

plot_b = px.line(s, x=ColumnsData.date, y=ColumnsData.cases, color=ColumnsData.province, template="plotly_white")
plot_b.update_layout(title='COVID-19 in Indonesia: total number of cases over time',
                      xaxis_title=ColumnsData.date, yaxis_title='Number of cases',
                      legend_title='<b>Top %s provinces</b>' % limit,
                      legend=dict(x=0.02, y=0.98))
plot_b.show()


# In[22]:



heatmap = data[(data[ColumnsData.cases] >= 100)].sort_values([ColumnsData.date, ColumnsData.province])
vis_hmap = go.Figure(data=go.Heatmap(
    z=heatmap[ColumnsData.cases],
    x=heatmap[ColumnsData.date],
    y=heatmap[ColumnsData.province],
    colorscale='Plasma'))

vis_hmap.update_layout(
    title='COVID-19 in Indonesia: number of cases over time', xaxis_nticks=20)

vis_hmap.show()


# In[23]:


corr = t.corr().iloc[[0, 1]].transpose()
corr = corr[(corr[ColumnsData.cases] > 0.25)].sort_values(ColumnsData.cases, ascending=False)
features = corr.index.tolist()
features.append(ColumnsData.mortality)
print('Selected features:', features)

d = t[features].copy()
d.head(10)


# In[29]:


create_bin(d, [
    ColumnsData.cases,
    ColumnsData.recovered,
    ColumnsData.density,
    ColumnsData.actives_cases,
    ColumnsData.deaths,
    ColumnsData.population,
    ColumnsData.mortality
], q=8)

normalisasi_data(d, d.columns)
d.head(20).T


# In[30]:


X = d[['Total Cases', 'Total Recovered', 'Population Density', 'Total Active Cases', 'Total Deaths', 'Population', 'Mortality']]  


# In[31]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#Elbow Method - Inertia plot
inertia = []
for k in range(1, 10):
    cluster_model = KMeans(n_clusters = k, random_state = 24)
    cluster_model.fit(X)
    inertia_value = cluster_model.inertia_
    inertia.append(inertia_value)
fig, ax = plt.subplots(figsize=(18, 16))
plt.plot(range(1, 10), inertia)
plt.title('The Elbow Method - Inertia plot', fontsize = 20)
plt.xlabel('No. of Clusters')
plt.ylabel('WCSS')
plt.show()


# In[34]:


X = d[['Total Cases', 'Total Recovered', 'Population Density', 'Total Active Cases', 'Total Deaths', 'Population', 'Mortality']] .values # Ambil value/nilai atribut Spending Score dan Annual Income
kmeans = KMeans(n_clusters=5, # Jumlah cluster K
                init='k-means++',  # Metode inisialisasi centroid
                random_state=111)


# In[39]:


y_kmeans = kmeans.fit_predict(X) # Fit dan prediksi dataset
y_kmeans


# In[40]:


plt.figure(1, figsize=(12, 5))
plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=50, c='red', label='Cluster 1')
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=50, c='blue', label='Cluster 2')
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s=50, c='green', label='Cluster 3')
plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], s=50, c='cyan', label='Cluster 4')
plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==4, 1], s=50, c='black', label='Cluster 5')


plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='purple', label='Centroids')
plt.title('Clusters of customers')
plt.xlabel('Province')
plt.ylabel('Deaths per case')
plt.legend()
plt.show()


# In[44]:


X = d[['Total Cases', 'Total Recovered', 'Population Density', 'Total Active Cases', 'Total Deaths', 'Population', 'Mortality']] .values # Ambil value/nilai atribut Spending Score dan Annual Income
kmeans = KMeans(n_clusters=6, # Jumlah cluster K
                init='k-means++',  # Metode inisialisasi centroid
                random_state=111)


# In[45]:


y_kmeans = kmeans.fit_predict(X) # Fit dan prediksi dataset
y_kmeans


# In[46]:


plt.figure(1, figsize=(12, 5))
plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=50, c='red', label='Cluster 1')
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=50, c='blue', label='Cluster 2')
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s=50, c='green', label='Cluster 3')
plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], s=50, c='cyan', label='Cluster 4')
plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==4, 1], s=50, c='black', label='Cluster 5')
plt.scatter(X[y_kmeans==5, 0], X[y_kmeans==5, 1], s=50, c='pink', label='Cluster 6')


plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='purple', label='Centroids')
plt.title('Clusters of customers')
plt.xlabel('Province')
plt.ylabel('Deaths per case')
plt.legend()
plt.show()


# In[48]:


kmeans = KMeans(n_clusters=6)
pred = kmeans.fit_predict(d[d.columns])
t['K-means Cluster Results'], d['K-means Cluster Results'] = [pred, pred]
d[d.columns].sort_values(['K-means Cluster Results', ColumnsData.mortality, 
                          ColumnsData.cases, ColumnsData.actives_cases, 
                          ColumnsData.density], ascending=True)


# In[51]:


vis_tmap = px.treemap(t.reset_index(), path=['K-means Cluster Results', ColumnsData.province], values=ColumnsData.cases)
vis_tmap.update_layout(title='K-means clusters untuk kasus di setiap provinsi')
vis_tmap.show()


# In[52]:


vis_tmap = px.treemap(t.reset_index(), path=['K-means Cluster Results', ColumnsData.province], values=ColumnsData.mortality)
vis_tmap.update_layout(title='K-means clusters untuk rata rata kematian di setiap provinsi')
vis_tmap.show()


# In[54]:


c = t.sort_values(['K-means Cluster Results', ColumnsData.cases], ascending=False)
data = [go.Bar(x=c[(c['K-means Cluster Results'] == i)].index, y=c[(c['K-means Cluster Results'] == i)][ColumnsData.cases],
               text=c[(c['K-means Cluster Results'] == i)][ColumnsData.cases], name=i) for i in range(0, 10)]

vis_bar = go.Figure(data=data)
vis_bar.update_layout(title='K-means Clustering: kasus di setiap provinsi',
                      xaxis_title='Indonesia State', yaxis_title='Deaths per case')
vis_bar.show()


# In[56]:


# visualization mortality rate by clusters
c = t.sort_values(['K-means Cluster Results', ColumnsData.mortality], ascending=False)
data = [go.Bar(x=c[(c['K-means Cluster Results'] == i)].index, y=c[(c['K-means Cluster Results'] == i)][ColumnsData.mortality],
               text=c[(c['K-means Cluster Results'] == i)][ColumnsData.mortality], name=i) for i in range(0, 10)]
data.append(
    go.Scatter(
        x=t.sort_values(ColumnsData.mortality, ascending=False).index,
        y=np.full((1, len(t.index)), 0.03).tolist()[0],
        marker_color='black',
        name='Indonesian avg'
    )
)

vis_bar2 = go.Figure(data=data)
vis_bar2.update_layout(title='K-means Clustering: rata rata kematian di setiap provinsi',
                       xaxis_title='Indonesian states', yaxis_title='Deaths per case')
vis_bar2.show()


# In[ ]:




