
# coding: utf-8

# # SF and Montréal weather using plotly and wunderground APIs

# This IPython notebook was prepared for this article in <em>Modern Data</em>: http://mod.plot.ly/update-plotly-charts-with-cron-jobs-and-python

# In[15]:

import urllib2
import json
import plotly
import datetime
import plotly.plotly as py
from plotly.graph_objs import *
py.sign_in('Python-Demo-Account', 'gwt101uhh0')


# In[1]:

api_key = 'XXXX' # Grab your own, free wunderground API key by signing up here: http://www.wunderground.com/weather/api 
sf_lookup_url = 'http://api.wunderground.com/api/' + api_key + '/geolookup/conditions/q/CA/San_Francisco.json'
mtl_lookup_url = 'http://api.wunderground.com/api/' + api_key + '/conditions/q/Canada/Montreal.json'


# ## Retrieve current temperature in Montréal and San Francisco 

# Code modified from: http://www.wunderground.com/weather/api/d/docs?d=resources/code-samples&MR=1

# In[7]:

urls = { 'MTL': mtl_lookup_url, 'SF': sf_lookup_url }
temps = { 'MTL': [], 'SF': [] }

for city in temps.keys():
    f = urllib2.urlopen(urls[city])
    json_string = f.read()
    parsed_json = json.loads(json_string)
    temps[city].append( parsed_json['current_observation']['temp_c'] )
    temps[city].append( parsed_json['current_observation']['temp_f'] )
    print "Current temperature in %s is: %s C, %s F" % (city, temps[city][0], temps[city][1] )
    f.close()


# ## Graph temperature data with Plotly Python client

# Get started with the Plotly Python client here: https://plot.ly/python/getting-started/

# ### Create a pretty chart layout object (the code block below is only for styling)

# (<strong>Tip</strong>: Its easiest to style charts in Plotly's online GUI, then copy-paste the styling code from the plot's "CODE" tab, ie https://plot.ly/~jackp/1837)

# In[24]:

layout = Layout(
    title='Current temperature in Montréal and San Francisco',
    titlefont=Font(
        family='"Open sans", verdana, arial, sans-serif',
        size=17,
        color='#444'
    ),
    font=Font(
        family='"Open sans", verdana, arial, sans-serif',
        size=12,
        color='#444'
    ),
    showlegend=True,
    autosize=True,
    width=803,
    height=566,
    xaxis=XAxis(
        title='Click to enter X axis title',
        titlefont=Font(
            family='"Open sans", verdana, arial, sans-serif',
            size=14,
            color='#444'
        ),
        range=[1418632334984.89, 1418632334986.89],
        domain=[0, 1],
        type='date',
        rangemode='normal',
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=True,
        autotick=True,
        nticks=0,
        ticks='inside',
        showticklabels=True,
        tick0=0,
        dtick=1,
        ticklen=5,
        tickwidth=1,
        tickcolor='#444',
        tickangle='auto',
        tickfont=Font(
            family='"Open sans", verdana, arial, sans-serif',
            size=12,
            color='#444'
        ),
        mirror='allticks',
        linecolor='rgb(34,34,34)',
        linewidth=1,
        anchor='y',
        side='bottom'
    ),
    yaxis=YAxis(
        title='Temperature (degrees)',
        titlefont=Font(
            family='"Open sans", verdana, arial, sans-serif',
            size=14,
            color='#444'
        ),
        range=[-5.968375815056313, 57.068375815056314],
        domain=[0, 1],
        type='linear',
        rangemode='normal',
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=True,
        autotick=True,
        nticks=0,
        ticks='inside',
        showticklabels=True,
        tick0=0,
        dtick=1,
        ticklen=5,
        tickwidth=1,
        tickcolor='#444',
        tickangle='auto',
        tickfont=Font(
            family='"Open sans", verdana, arial, sans-serif',
            size=12,
            color='#444'
        ),
        exponentformat='B',
        showexponent='all',
        mirror='allticks',
        linecolor='rgb(34,34,34)',
        linewidth=1,
        anchor='x',
        side='left'
    ),
    legend=Legend(
        x=1.00,
        y=1.02,
        traceorder='normal',
        font=Font(
            family='"Open sans", verdana, arial, sans-serif',
            size=12,
            color='#444'
        ),
        bgcolor='rgba(255, 255, 255, 0.5)',
        bordercolor='#444',
        borderwidth=0,
        xanchor='left',
        yanchor='auto'
    )
)


# ### Graph the temperature data

# In[25]:

cur_time = datetime.datetime.now() # current date and time
data=[]
temp_types = ['C','F']
for city in temps.keys():
    for i in range(len(temp_types)):
        data.append( Scatter( x=[cur_time], y=[temps[city][i]],                              line=Line(dash='dot') if i==0 else Line(),
                             mode='lines+markers', \
                             name='{0} ({1})'.format(city,temp_types[i]) ) )

data = Data( data )
fig = Figure(data=data, layout=layout)
py.iplot(fig, filename='montreal-and-san-francisco-temperatures')


# <strong>Notes on automating this script</strong>
# 
# This Python script can be automated using cron jobs (https://help.ubuntu.com/community/CronHowto) and setting the "fileopt" kwarg in py.plot() to "extend" (https://plot.ly/python/file-options/). Also, make sure that you set the "auto_open" kwarg to False, so that your server does not try to open your plot as an HTML page (https://plot.ly/python/overview/).
# 
# In your Python script, replace the py.iplot() call above with:
# <code>py.plot(fig, filename='montreal-and-san-francisco-temperatures', fileopt='extend', auto_open=False)</code>. You can copy the full Python script from here: https://gist.github.com/jackparmer/9de837d2ccc24b483a2e#file-graph-weather-underground-data-with-plotly-python-client
# 
# Finally, make sure to include <code>#!/usr/bin/env python</code> as the first line at the top of your Python script. Cron will not be able to run your Python script if this is not the first line.
