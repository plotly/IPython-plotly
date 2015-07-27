
# coding: utf-8

# In[1]:

import plotly.plotly as py
from plotly.graph_objs import *


# In[2]:

import pandas as pd


# In[3]:

import numpy as np # for generating random data


# ## Plotly with Pandas
# 
# Plotly's Python library fully supports Pandas series. Instead of passing a `list` to `x` and `y`, just pass a pandas series (column) or index.
# 
# This is a simple primer on basic plotting with basic dataframes. For more on chart types and customization, [view our python documentation.](https://plot.ly/python/)
# 
# - Plotting Basic Chart Types with Pandas Data Frames
# - Editing Axes Labels and Titles
# - Saving Graphs as Images to File
# 
# 
# ### More resources
# - [Plotly's Python documentation](https://plot.ly/python/)
# - [A great notebook on the medium data Pandas, SQLite, and Plotly workflow](https://plot.ly/ipython-notebooks/big-data-analytics-with-pandas-and-sqlite/)
# - Questions? <support@plot.ly>, [@plotlygraphs](https://twitter.com/plotlygraphs)

# ## Plotting Basic Chart Types
# ## with Pandas Dataframes
# #### See more examples of creating and customizing plotly figures in our [python documentation](https://plot.ly/python)

# In[4]:

N = 500
x = np.linspace(0, 1, N)
y = np.random.randn(N)
df = pd.DataFrame({'x': x, 'y': y})
df.head()


# In[5]:

data = [
    Scatter(
        x=df['x'], # assign x as the dataframe column 'x'
        y=df['y']
    )
]
py.iplot(data, filename='pandas/example 1')


# In[1]:

# Or plot the index
data = [
    Scatter(
        x=df.index, # assign x as the dataframe index
        y=df['y'],
        mode='markers'
    )
]
py.iplot(data, filename='pandas/example 1 - index')


# In[6]:

data = [
    Scatter(
        x=df['x'], # assign x as the dataframe column 'x'
        y=df['y'],
        mode='markers'
    )
]
py.iplot(data, filename='pandas/example 2')


# In[7]:

data = [
    Bar(
        x=df['x'], # assign x as the dataframe column 'x'
        y=df['y']
    )
]
py.iplot(data, filename='pandas/example 3')


# In[8]:

data = [
    Histogram2dContour(
        x=df['x'], # assign x as the dataframe column 'x'
        y=df['y']
    ),
    Scatter(
        x=df['x'], # assign x as the dataframe column 'x'
        y=df['y'],
        mode='markers'
    ),    
]
py.iplot(data, filename='pandas/example 4')


# In[11]:

data = [
    Histogram(
        y=df['y']
    )    
]
py.iplot(data, filename='pandas/example 5')


# In[13]:

data = [
    Box(
        y=df['y']
    )    
]
py.iplot(data, filename='pandas/example 6')


# ## Adding Titles and Labels

# In[14]:

data = [
    Scatter(
        x=df['x'], # assign x as the dataframe column 'x'
        y=df['y']
    )
]

layout = Layout(
    title='scatter plot with pandas',
    yaxis=YAxis(title='random distribution'),
    xaxis=XAxis(title='linspace')
)

fig = Figure(data=data, layout=layout)

py.iplot(fig, filename='pandas/example 7')


# ## Graphing Multiple Series

# In[15]:

x = np.linspace(0, 1, N)
y = np.random.randn(N) + 3
df2 = pd.DataFrame({'x': x, 'y': y})
df2.head()


# In[16]:

data = [
    Scatter(
        x=df['x'], # assign x as the dataframe column 'x'
        y=df['y'],
        name='random around 0'
    ),
    Scatter(
        x=df2['x'], # assign x as the dataframe column 'x'
        y=df2['y'],
        name='random around 3'
    )    
]

py.iplot(data, filename='pandas/example 8')


# In[17]:

data = [
    Bar(
        x=df['x'], # assign x as the dataframe column 'x'
        y=df['y'],
        name='random around 0'
    ),
    Scatter(
        x=df2['x'], # assign x as the dataframe column 'x'
        y=df2['y'],
        name='random around 3'
    )    
]

py.iplot(data, filename='pandas/example 9')


# ## Saving images

# In[19]:

py.image.save_as(data, filename='pandas-example-9.png')


# In[20]:

ls *.png


# In[21]:

from IPython.display import Image
Image(filename='pandas-example-9.png') 


# In[2]:

from IPython.core.display import HTML
import urllib2
HTML(urllib2.urlopen('https://raw.githubusercontent.com/plotly/python-user-guide/css-updates/custom.css').read())


# In[ ]:



