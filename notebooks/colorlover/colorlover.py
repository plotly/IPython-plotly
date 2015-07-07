
# coding: utf-8

# ## colorlover

# ##### Color scales for IPython notebook

# ###### Install with pip, <code>sudo pip install colorlover</code> Code and documentation on <a href="https://github.com/jackparmer/colorlover" target="_blank">Github</a>

# In[2]:

import colorlover as cl
from IPython.display import HTML


# ##### Display a single color scale

# In[3]:

HTML(cl.to_html( cl.scales['3']['div']['RdYlBu'] ))


# ##### Display many color scales

# In[19]:

HTML(cl.to_html( cl.scales['11'] )) # All scales with 11 colors


# ##### Display sequential color scales (with 3 colors)

# In[14]:

HTML(cl.to_html( cl.flipper()['seq']['3'] ))


# ##### Scales are RGB by default...

# In[15]:

ryb = cl.scales['3']['div']['RdYlBu']; ryb


# ##### But its easy to change to HSL...

# In[17]:

cl.to_hsl( ryb )


# ##### Or tuples of RGB values

# In[18]:

cl.to_numeric( ryb )


# #### Color interpolation

# In[43]:

bupu = cl.scales['9']['seq']['BuPu']
buHTML( cl.to_html(bupu) )


# In[49]:

bupu500 = cl.interp( bupu, 500 ) # Map color scale to 500 bins
HTML( cl.to_html( bupu500 ) )


# #### Creating plots

# ###### (pip install --upgrade plotly for latest Plotly package version)

# In[62]:

import plotly.plotly as py
from plotly.graph_objs import *
import math

un='IPython.Demo'; k='1fw3zw2o13'; py.sign_in(un,k);

data = Data([ Scatter(
    x = [ i * 0.1 for i in range(500) ],
    y = [ math.sin(j * 0.1) for j in range(500) ],
    mode='markers',
    marker=Marker(color=bupu500,size=22.0,line=Line(color='black',width=2)),
    text = cl.to_rgb( bupu500 ),
    opacity = 0.7
)])
layout = Layout( showlegend=False, xaxis=XAxis(zeroline=False), yaxis=YAxis(zeroline=False) )
fig = Figure(data=data, layout=layout)
py.iplot(fig, filename='spectral_bubblechart')


# #### All colors

# In[20]:

HTML(cl.to_html( cl.scales ))


# Color scales in <code>cl.scales</code> and much inspiration are from <a href="colorbrewer.org">ColorBrewer</a>
