
# coding: utf-8

# #Cufflinks
# 
# This library binds the power of [plotly](http://www.plot.ly) with the flexibility of [pandas](http://pandas.pydata.org/) for easy plotting.
# 
# This library is available on https://github.com/santosjorge/cufflinks
# 
# This tutorial assumes that the plotly user credentials have already been configured as stated on the [getting started](https://plot.ly/python/getting-started/) guide.

# In[4]:

import cufflinks as cf


# ## Line Chart

# In[6]:

cf.datagen.lines().iplot(kind='scatter',xTitle='Dates',yTitle='Returns',title='Cufflinks - Line Chart',
                         world_readable=True)


# In[29]:

cf.datagen.lines(3).iplot(kind='scatter',xTitle='Dates',yTitle='Returns',title='Cufflinks - Filled Line Chart',
                         colorscale='-blues',fill=True,world_readable=True)


# In[8]:

cf.datagen.lines(1).iplot(kind='scatter',xTitle='Dates',yTitle='Returns',title='Cufflinks - Besfit Line Chart',
                         filename='Cufflinks - Bestfit Line Chart',bestfit=True,colors=['blue'],
                         bestfit_colors=['pink'],world_readable=True)


# ## Scatter Chart

# In[9]:

cf.datagen.lines(2).iplot(kind='scatter',mode='markers',size=10,symbol='x',colorscale='paired',
                          xTitle='Dates',yTitle='EPS Growth',title='Cufflinks - Scatter Chart',
                          world_readable=True)


# ## Spread Chart

# In[19]:

cf.datagen.lines(2).iplot(kind='spread',xTitle='Dates',yTitle='Return',title='Cufflinks - Spread Chart',
                          world_readable=True)


# ## Bar Chart

# In[20]:

cf.datagen.lines(5).resample('M').iplot(kind='bar',xTitle='Dates',yTitle='Return',title='Cufflinks - Bar Chart',
                          world_readable=True)


# In[21]:

cf.datagen.lines(5).resample('M').iplot(kind='bar',xTitle='Dates',yTitle='Return',title='Cufflinks - Grouped Bar Chart',
                          barmode='stack',world_readable=False)


# ## Box Plot

# In[22]:

cf.datagen.box(6).iplot(kind='box',xTitle='Stocks',yTitle='Returns Distribution',title='Cufflinks - Box Plot',
                        world_readable=True)


# ## Historgram

# In[23]:

cf.datagen.histogram(2).iplot(kind='histogram',opacity=.75,title='Cufflinks - Histogram',
                              linecolor='white',world_readable=True)


# ##Heatmap Plot

# In[24]:

cf.datagen.heatmap(20,20).iplot(kind='heatmap',colorscale='spectral',title='Cufflinks - Heatmap',
                                world_readable=True)


# ##Bubble Chart

# In[25]:

cf.datagen.bubble(prefix='industry').iplot(kind='bubble',x='x',y='y',size='size',categories='categories',text='text',
                          xTitle='Returns',yTitle='Analyst Score',title='Cufflinks - Bubble Chart',
                          world_readable=True)


# ##Scatter 3D

# In[26]:

cf.datagen.scatter3d(2,150).iplot(kind='scatter3d',x='x',y='y',z='z',size=15,categories='categories',text='text',
                             title='Cufflinks - Scatter 3D Chart',colors=['blue','pink'],width=0.5,margin=(0,0,0,0),
                             world_readable=True)


# ##Bubble 3D 

# In[27]:

cf.datagen.bubble3d(5,4).iplot(kind='bubble3d',x='x',y='y',z='z',size='size',text='text',categories='categories',
                            title='Cufflinks - Bubble 3D Chart',colorscale='set1',
                            width=.5,opacity=.8,world_readable=True)


# ##Surface

# In[28]:

cf.datagen.sinwave(10,.25).iplot(kind='surface',theme='solar',colorscale='brbg',title='Cufflinks - Surface Plot',
                                 margin=(0,0,0,0),world_readable=True)


# In[28]:




# In[ ]:



