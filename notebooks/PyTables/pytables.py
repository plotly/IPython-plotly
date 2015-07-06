
# coding: utf-8

# ### HDF5 and Plotly

# This notebook will give an overview of using the excellent [HDF5 Data Format](https://www.hdfgroup.org/HDF5/) for high performance computing and [Plotly](https://plot.ly/) to graph data stored in this files.
#  Plotly is a web-based graphing platform that lets you make and share interactive graphs and dashboards. You can use it for free online--sign up for an account [here](https:www.plot.ly)--and on-premise with [Plotly Enterprise](https://plot.ly/product/enterprise/).
# 
# 
# For those unfamilar with the HDF5 file format:
# 
# 
# 
# HDF5 is a data model, library, and file format for storing and managing data. It supports an unlimited variety of datatypes, and is designed for flexible and efficient I/O and for high volume and complex data. HDF5 is portable and is extensible, allowing applications to evolve in their use of HDF5. The HDF5 Technology suite includes tools and applications for managing, manipulating, viewing, and analyzing data in the HDF5 format.
# 
# 
# 
# -- [The HDF5 Group](https://www.hdfgroup.org/HDF5/)
# 
# 
# 
# The HDF group has some great reasons to use their files - namely that it works great with all kind of data. You can [read more here.](https://www.hdfgroup.org/why_hdf/)

# In[1]:

import pandas as pd
from IPython.display import display
import plotly.plotly as py # interactive graphing
from plotly.graph_objs import Bar, Scatter, Marker, Layout, Data, Figure, Heatmap, XAxis, YAxis
import plotly.tools as tls
import numpy as np


# The dataset that we'll be using is data from [NYC's open data portal](https://nycopendata.socrata.com/data). We'll be exploring a 100mb dataset covering traffic accidents in NYC. While we are capable of fitting this data into memory, the HDF5 file format has some unique affordances that allow us to query and save data in convenient ways.

# Now the first thing we'll want to do is open up an access point to this HDF5 file, doing so is simple because pandas provides ready access to doing so.

# In[2]:

pd.set_option('io.hdf.default_format','table')


# In[3]:

store = pd.HDFStore('nypd_motors.h5')


# Now that we've opened up our store, let's start storing some data

# In[4]:

# df = pd.read_csv('NYPD_motor_collisions.csv', parse_dates=['DATE'])
# df.columns = [col.lower().replace(" ", "_") for col in df.columns]
# store.append("nypd", df,format='table',data_columns=True)


# In[5]:

store


# In[6]:

# store.close()


# One thing that's nice about the HDF5 file is that it's kind of like a key value store. It's simple to use, and allows you to store things just like you might in a file system type hierarchy.

# What's awesome about the HDF5 format is that it's almost like a miniature file system. It supports hierarchical data and is accessed like a python dictionary.

# In[7]:

store.get_storer("df")


# In[8]:

store.select("nypd").head()


# In[9]:

boroughs = store.select("nypd", "columns=['borough']")


# In[10]:

boroughs['COUNT'] = 1
borough_groups = boroughs.groupby('borough')


# In[11]:

borough_groups.sum().index


# In[12]:

data = Data([Bar(y=borough_groups.sum()['COUNT'], x=borough_groups.sum().index)])
layout = Layout(xaxis=XAxis(title="Borough"), yaxis=YAxis(title='Accident Count'))
fig = Figure(data=data, layout=layout)


# In[13]:

py.iplot(fig)


# In[14]:

dates_borough = store.select("nypd", "columns=['date', 'borough']").sort('date')


# In[15]:

dates_borough['COUNT'] = 1


# In[16]:

date_borough_sum = dates_borough.groupby(['borough', "date"]).sum()
date_borough_sum.head()


# In[17]:

data = []
for g, df in date_borough_sum.reset_index().groupby('borough'):
    data.append(Scatter(x= df.date, y=df.COUNT,name=g))
layout = Layout(xaxis=XAxis(title="Date"), yaxis=YAxis(title="Accident Count"))


# In[18]:

py.iplot(Figure(data=Data(data), layout=layout), filename='nypd_crashes/over_time')


# Luckily for us, while this graph is a bit of a mess, we can still zoom in on specific times and ranges. This makes plotly perfect for exploring datasets. You can create a high level visual of the data then zoom into a more detailed level.
# 
# See below where using the above graph I could zoom in on a particular point and anontate it for future investigation.

# In[19]:

tls.embed("https://plot.ly/~bill_chambers/274")


# In[20]:

car_types = store.select("nypd", "columns=['vehicle_type_code_1', 'vehicle_type_code_2']")
car_types['COUNT'] = 1


# In[21]:

code_1 = car_types.groupby('vehicle_type_code_1').sum()
code_2 = car_types.groupby('vehicle_type_code_2').sum()


# In[22]:

data = Data([
        Bar(x=code_1.index, y=code_1.COUNT,name='First Vehicle Type'),
        Bar(x=code_2.index, y=code_2.COUNT,name='Second Vehicle Type')
     ])


# In[23]:

py.iplot(Figure(data=data, layout=Layout(barmode='group', yaxis=YAxis(title="Vehicle Incidents"))))


# No big surprises here, we can see that passenger vehicles, likely being the most prevalent vehicles, are the ones involved in the most accidents for the first and second vehicles. However this does make for some more interesting questions, does this extrapolate to each vehicle class. That is, do all kinds of vehicles hit all other vehicles in more or less the same frequency? 
# 
# Let's explore large commercial vehicles.

# In[24]:

large_vehicles = car_types.groupby(
    'vehicle_type_code_1'
).get_group(
    'LARGE COM VEH(6 OR MORE TIRES)'
).groupby('vehicle_type_code_2').sum()


# In[25]:

data = Data([Bar(x=large_vehicles.index,y=large_vehicles.COUNT)])
py.iplot(Figure(data=data, layout=Layout(yaxis=YAxis(title="Incident Per Vehicle Type"))))


# At first glance it seems alright, but it's worth more exploration - let's Z-Score the data and compare their scores.

# In[26]:

large_vehicles.head()


# In[27]:

code_2.head()


# In[28]:

def z_score(df):
    df['zscore'] = ((df.COUNT - df.COUNT.mean())/df.COUNT.std())
    return df


# In[29]:

data = Data([
        Bar(x=z_score(code_2).index,y=z_score(code_2).zscore, name='All Vehicles'),
        Bar(x=z_score(large_vehicles).index,y=z_score(large_vehicles).zscore,name='Large Vehicles'),
        
     ])


# In[30]:

py.iplot(Figure(data=data, layout=Layout(yaxis=YAxis(title="Incident Per Vehicle Type"))),name='nypd_crashes/large vs all vehicles')


# We can see that things are relatively similar, except that large vehicles seem to hit large vehicles much more than most others. This could warrant further investigation.
# 
# While grouped bar charts can be useful for these kinds of comparisons, it can be great to visualize this data with heatmaps as well. We can create one of these by creation a contingency table or cross tabulation.

# In[31]:

cont_table = pd.crosstab(car_types['vehicle_type_code_1'], car_types['vehicle_type_code_2']).apply(np.log)


# Because of the different magnitudes of data, I decided to log scale it.

# In[32]:

py.iplot(Data([
            Heatmap(z = cont_table.values, x=cont_table.columns, y=cont_table.index, colorscale='Jet')
        ]),filename='nypd_crashes/vehicle to vehicle heatmap')


# With this we are able to see more interesting nuances in the data. For instance taxis seems to have lots of accidents with other taxis, while vans and station wagons also seem to have many accidents.
# 
# There's clearly a lot to explore in this dataset.
