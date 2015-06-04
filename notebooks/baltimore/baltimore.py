
# coding: utf-8

# 
# # Baltimore Vital Signs

# The repo for this notebook lives [here](https://github.com/mkcor/baltimore_vital_signs). It was forked from [https://github.com/jtelszasz/baltimore_vital_signs](https://github.com/jtelszasz/baltimore_vital_signs).

# The <a href='http://bniajfi.org/indicators/all'>Baltimore Neighborhoods Indicators Alliance - Jacob France Institute (BNIA)</a> at the University of Baltimore has made it their mission to provide a clean, concise set of indicators that illustrate the health and wealth of the city. There are 152 socio-economic indicators in the Vital Signs dataset, and some are reported for multiple years which results in 295 total variables for each of the 56 Baltimore neighborhoods captured.  The indicators are dug up from a number of sources, including the U.S. Census Bureau and its American Community Survey, the FBI and Baltimore Police Department, Baltimore departments of city housing, health, and education.

# In[1]:

import glob
import numpy as np
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as pgo


# In[2]:

# Load and combine the datasets.
path = 'raw_data/csv'

allFiles = glob.glob(path + '/*.csv')
df = pd.DataFrame()

for i, filename in enumerate(allFiles):
    df_file = pd.read_csv(filename)
    if i == 0:
        df = df_file
    else:
        df = pd.merge(df, df_file)


# In[3]:

df.index = df['CSA2010']
df.drop('CSA2010', inplace=True)
print len(df.columns)
del df['CSA2010']
print len(df.columns)


# In[4]:

cols = df.columns
df[cols] = (
    df[cols]
    # Replace things that aren't numbers and change any empty entries to nan
    # (to allow type conversion)
    .replace({r'[^0-9\.]': '', '': np.nan}, regex=True)
    # Change to float and convert from %s
    .astype(np.float64)
)


# In[5]:

# One of the rows is an aggregate Baltimore City.
df.drop('Baltimore City', inplace=True)


# # A Few Exploratory Plots

# ## Percentage of Population White in Each Neighborhood, Sorted

# In[6]:

df_white_sorted = df['pwhite10'].sort(inplace=False)


# In[7]:

# Create a horizontal bar chart with plotly.
data = pgo.Data([
    pgo.Bar(
            y=df_white_sorted.index,
            x=df_white_sorted,
            orientation='h'
    )
])


# In[8]:

layout = pgo.Layout(
    title='% White',
    margin=pgo.Margin(l=300)  # add left margin for y-labels are long
)


# In[9]:

fig = pgo.Figure(data=data, layout=layout)


# In[10]:

# Address InsecurePlatformWarning from running Python 2.7.6
import urllib3.contrib.pyopenssl
urllib3.contrib.pyopenssl.inject_into_urllib3()


# In[11]:

py.iplot(fig, filename='baltimore-barh',
         width=700, height=1000)  # adjust notebook display width and height


# ## Percentage of Households in Poverty and with Children, Sorted

# In[12]:

df_chpov_sorted = df['hhchpov12'].sort(inplace=False)


# In[13]:

data1 = pgo.Data([
    pgo.Bar(
            y=df_chpov_sorted.index,
            x=df_chpov_sorted,
            orientation='h'
    )
])


# In[14]:

# Specify some layout attributes.
layout1 = pgo.Layout(
    title='% HH w. Children in Poverty',
    margin=pgo.Margin(l=300)  # add left margin for y-labels are long
)


# In[15]:

fig1 = pgo.Figure(data=data1, layout=layout1)


# In[16]:

py.iplot(fig1, filename='baltimore-hh-pov', width=700, height=1000)


# ## Percentage Households in Poverty with Children vs Percentage Population White (per Neighborhood)

# ### Bubbles Sized by Juvenile Population (per Neighborhood)

# In[17]:

# Juvenile population (age 10 to 18)
juv_pop = df['tpop10'] * df['age18_10'] / 100


# In[18]:

# Display this information in hover box.
hover_text = zip(juv_pop.index, np.around(juv_pop, 2))


# In[19]:

# Represent a third dimension (size).
data2 = pgo.Data([
    pgo.Scatter(
            x=df['pwhite10'],
            y=df['hhchpov12'],
            mode='markers',
            marker=pgo.Marker(size=juv_pop,
                              sizemode='area',
                              sizeref=juv_pop.max()/600,
                              opacity=0.4,
                              color='blue'),
            text=hover_text
    )
])


# In[20]:

layout2 = pgo.Layout(
    title='Baltimore: Too Many Non-White Kids in Poverty',
    xaxis=pgo.XAxis(title='% Population White (2010)',
                    range=[-5, 100],
                    showgrid=False,
                    zeroline=False),
    yaxis=pgo.YAxis(title='% HH w. Children in Poverty (2012)',
                    range=[-5, 100],
                    showgrid=False,
                    zeroline = False),
    hovermode='closest'
)


# In[21]:

fig2 = pgo.Figure(data=data2, layout=layout2)


# In[22]:

py.iplot(fig2, filename='baltimore-bubble-chart')


# ## Percentage of Households in Poverty vs Ethnicity's Percentage of Population

# Let's do this chart using matplotlib for a change.

# In[23]:

import matplotlib.pyplot as plt


# In[24]:

mpl_fig, ax = plt.subplots()

size = 100
alpha = 0.5
fontsize = 16

ax.scatter(df['phisp10'], df['hhpov12'], c='r', alpha=alpha, s=size)
ax.scatter(df['paa10'], df['hhpov12'], c='c', alpha=alpha, s=size)
ax.legend(['Hispanic', 'Black'], fontsize=12)

# Turn off square border around plot.
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Turn off ticks.
ax.tick_params(axis="both", which="both", bottom="off", top="off",
               labelbottom="on", left="off", right="off", labelleft="on",
               labelsize=16)

ax.set_ylim(-5, 60)
ax.set_xlim(-5, 100)

ax.set_ylabel('% HH in Poverty', fontsize=fontsize)
ax.set_xlabel('% Population', fontsize=fontsize)


# Matplotlib code is very long... But sometimes you have existing matplotlib code, right? The good news is, plotly can eat it! 

# In[25]:

py.iplot_mpl(mpl_fig, filename='baltimore-poverty')


# So, at the moment, matplotlib legends do not fully convert to plotly legends (please refer to our [user guide](https://plot.ly/python/matplotlib-to-plotly-tutorial/#Careful,-matplotlib-is-not-perfect-%28yet%29)). Let's tweak this now.

# In[26]:

import plotly.tools as tls


# In[27]:

# Convert mpl fig object to plotly fig object, resize to plotly's default.
py_fig = tls.mpl_to_plotly(mpl_fig, resize=True)


# In[28]:

# Give each trace a name to appear in legend.
py_fig['data'][0]['name'] = py_fig['layout']['annotations'][0]['text']
py_fig['data'][1]['name'] = py_fig['layout']['annotations'][1]['text']


# In[29]:

# Delete misplaced legend annotations. 
py_fig['layout'].pop('annotations', None)


# In[30]:

# Add legend, place it at the top right corner of the plot.
py_fig['layout'].update(
    showlegend=True,
    legend=pgo.Legend(
        x=1,
        y=1
    )
)


# In[31]:

# Send updated figure object to Plotly, show result in notebook.
py.iplot(py_fig, filename='baltimore-poverty')


# Hispanic communities are smaller fractions of neighborhood populations.

# # Principal Component Analysis

# Read this [post](http://www.thetrainingset.com/articles/A-City-Divided-In-N-Dimensions) at The Training Set for purpose of the following analyses (this section and the next one).

# In[32]:

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[33]:

X = np.array(df)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[34]:

pca = PCA()
pca.fit(X_scaled)


# In[35]:

len(pca.components_)


# 55 dimensions (or components, or axes) were used in the Principal Component Analysis.

# In[36]:

print 'Explained Variance Ratio = ', sum(pca.explained_variance_ratio_[: 2])


# We can see that almost half (~48%) of the total variance comes from only two dimensions (i.e., the first two principal components). Let's visualize the relative contribution of all components.

# In[37]:

data3 = pgo.Data([
    pgo.Bar(
            y=pca.explained_variance_ratio_,
    )
])


# In[38]:

py.iplot(data3, filename="baltimore-principal-dimensions")


# Let's plot a cumulative version of this, to see how many dimensions are needed to account for 90% of the total variance.

# In[39]:

data4 = pgo.Data([
    pgo.Scatter(
            y=np.cumsum(pca.explained_variance_ratio_),
    )
])


# In[40]:

py.iplot(data4, filename='baltimore-pca-cumulative')


# So we need about 20 dimensions to explain ~90% of the total variance.

# Let's focus on the 2 principal dimensions, so it's easy to plot them in the (x, y) plane.

# In[41]:

pca.n_components = 2
X_reduced = pca.fit_transform(X_scaled)
df_X_reduced = pd.DataFrame(X_reduced, index=df.index)


# In[42]:

trace = pgo.Scatter(x=df_X_reduced[0],
                    y=df_X_reduced[1],
                    text=df.index,
                    mode='markers',
                    # Size by total population of each neighborhood. 
                    marker=pgo.Marker(size=df['tpop10'],
                                      sizemode='diameter',
                                      sizeref=df['tpop10'].max()/50,
                                      opacity=0.5)
)

data5 = pgo.Data([trace])


# In[43]:

layout5 = pgo.Layout(title='Baltimore Vital Signs (PCA)',
                     xaxis=pgo.XAxis(showgrid=False,
                                     zeroline=False,
                                     showticklabels=False),
                     yaxis=pgo.YAxis(showgrid=False,
                                     zeroline=False,
                                     showticklabels=False),
                     hovermode='closest'
)


# In[44]:

fig5 = pgo.Figure(data=data5, layout=layout5)
py.iplot(fig5, filename='baltimore-2dim')


# We have reduced a high-dimensional problem to a simple model. We can visualize it in 2 dimensions. Neighborhoods which lie closer to one another are more similar (with respect to these 'vital signs', i.e., socio-economic indicators). Downtown seems very special!

# # K-means Clustering

# Could we identify groups of similar neighborhoods? Clearly, Downtown forms its own group. It's not as easy to identify visually the other groups (or clusters). K-means clustering is an algorithmic method to compute closer data points (belonging to the same cluster), given the number of clusters you want.

# In[45]:

from sklearn.cluster import KMeans


# The total number of clusters you expect should be small enough (otherwise there's no *clustering*) but large enough so that *inertia* can be reasonable (small enough). Inertia measures the typical distance between a data point and the center of its cluster. 

# In[46]:

# Let the number of clusters be a parameter, so we can get a feel for an appropriate
# value thereof.
def cluster(n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X_reduced)
    Z = kmeans.predict(X_reduced)
    return kmeans, Z


# In[47]:

max_clusters = len(df)
# n_clusters = max_clusters would be trivial clustering.


# In[48]:

inertias = np.zeros(max_clusters)

for i in xrange(1, max_clusters):
    kmeans, Z = cluster(i)
    inertias[i] = kmeans.inertia_


# In[49]:

data6 = pgo.Data([
    pgo.Scatter(
            x=range(1, max_clusters),
            y=inertias[1:]
    )
])


# In[50]:

layout6 = pgo.Layout(
    title='Baltimore dataset - Investigate k-means clustering',
    xaxis=pgo.XAxis(title='Number of clusters',
                    range=[0, max_clusters]),
    yaxis=pgo.YAxis(title='Inertia')
)


# In[51]:

fig6 = pgo.Figure(data=data6, layout=layout6)


# In[52]:

py.iplot(fig6, filename='baltimore-clustering-inertias')


# Okay, let's go for 7 clusters.

# In[53]:

n_clusters = 7
model, Z = cluster(n_clusters)


# In[54]:

# Represent neighborhoods as in previous bubble chart, adding cluster information under color.
trace0 = pgo.Scatter(x=df_X_reduced[0],
                     y=df_X_reduced[1],
                     text=df.index,
                     name='',
                     mode='markers',
                     marker=pgo.Marker(size=df['tpop10'],
                                       sizemode='diameter',
                                       sizeref=df['tpop10'].max()/50,
                                       opacity=0.5,
                                       color=Z),
                     showlegend=False
)


# In[55]:

# Represent cluster centers.
trace1 = pgo.Scatter(x=model.cluster_centers_[:, 0],
                     y=model.cluster_centers_[:, 1],
                     name='',
                     mode='markers',
                     marker=pgo.Marker(symbol='x',
                                       size=12,
                                       color=range(n_clusters)),
                     showlegend=False
)


# In[56]:

data7 = pgo.Data([trace0, trace1])


# In[57]:

layout7 = layout5


# In[58]:

layout7['title'] = 'Baltimore Vital Signs (PCA and k-means clustering with 7 clusters)'


# In[59]:

fig7 = pgo.Figure(data=data7, layout=layout7)


# In[60]:

py.iplot(fig7, filename='baltimore-cluster-map')


# Zoom, pan, and hover to explore this reduction of the Baltimore Vital Signs dataset!
