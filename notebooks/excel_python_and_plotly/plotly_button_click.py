from IPython.display import IFrame
#A few imports we will need later
from xlwings import Workbook, Sheet, Range, Chart
import pandas as pd
import numpy as np
import plotly.plotly as py
import plotly.tools as tlsM
from IPython.display import HTML
from plotly.graph_objs import *


#workbook connection
wb = Workbook.caller()

#Now we can use some of these controls to customize the 
folder_name = Range('Dashboard','B2').value
graph_title = Range('Dashboard','B3').value


#short function to create a new dataframe using xlwings
def new_df(shtnm, startcell = 'A1'):
    data = Range(shtnm, startcell).table.value
    temp_df = pd.DataFrame(data[1:], columns = data[0])
    return(temp_df)

###Make some dataframes from the workbook sheets
#Core Product
shtnm1 = Range('Dashboard','B6').value
df = new_df(shtnm1)

#2nd Product
product_2 = False
if Range('Dashboard','C7').value == "Yes":
    shtnm2 = Range('Dashboard','B7').value
    df2 = new_df(shtnm2)
    product_2 = True

#3rd Product
product_3 = False
if Range('Dashboard','C8').value == "Yes":
    shtnm3 = Range('Dashboard','B8').value
    df3 = new_df(shtnm3)
    product_3 = True       


 #Clean up the charaters in the columns 
names2 = []
def clean_names(column_list):
    #Short function to make our column headers easier to reference later.
    names2=[]
    for name in column_list:
        name = name.replace(" ","").lower()
        if 'windowprice' in name: #this allows non-adult ticket types to be built
            name = 'windowprice'
        names2.append(name)
    return names2

df.columns = clean_names(df.columns.values)
if product_2 == True:
    df2.columns = clean_names(df2.columns.values)
if product_3 == True:
    df3.columns = clean_names(df3.columns.values)  


df= df.set_index('date').tz_localize('MST').astype(float)
if product_2 == True:
    df2= df2.set_index('date').tz_localize('MST').astype(float)
if product_3 == True:
    df3= df3.set_index('date').tz_localize('MST').astype(float)



#set a few global variables so we can use them throughout the plots
X = df.index

try:  
    ymin = min(df['minpriceoffered'].min(),df2['minpriceoffered'].min(),df3['minpriceoffered'].min()) - 10
    ymax = max(df['walkupprice'].max(),df2['walkupprice'].max(),df3['walkupprice'].max()) + 10
    
except:
    #If that doesn't work, just go edit it on Plotly's web based plot editor. 
    ymin = df['minpriceoffered'].min() - 10
    ymax = df['walkupprice'].max() + 10



#function to create a "trace" (line) for each item we want to plot
def new_trace(price_column, color, name, x=X, fill = 'none', qty_column = []):
    trace = Scatter(
    x=X,
    y=price_column,  
    fill=fill,
    mode='lines',
    name=name,
    text=['Quantity: {}'.format(q) for q in qty_column],
    line=Line(
        color=color,
        width=2,
        dash='solid',
        opacity=1,),
    xaxis='x1',
    yaxis='y1')
    return trace

#Set up the 3 core traces
trace1 = new_trace(df['walkupprice'], '#FF9966','Core Product Walkup Price') 
trace2 = new_trace(df['maxpriceoffered'], '#5EA5D1',shtnm1 + 'Highest Price Offered', qty_column=df['unitsmax']) 
trace3 = new_trace(df['minpriceoffered'], '#5EA5D1',shtnm1+' Starting Price',  qty_column= df['unitsmin'], fill='tonexty') 
trace_list = [trace1, trace2, trace3]




#add additional traces if toggled on by user
if product_2 == True:
    trace4 = new_trace(df2['minpriceoffered'], '##66ff66',shtnm2+' Lowest Price Offered')
    trace_list.append(trace4)  

if product_3 == True:
    trace5 = new_trace(df3['minpriceoffered'], '#e6e600',shtnm3+' Lowest Price Offered') 
    trace_list.append(trace5) 


layout = Layout(
    title=graph_title,                #Using the input from the Dashboard Sheet in Excel
    titlefont=Font(
        size=12.0,
        color='#262626'
    ),
    showlegend=True,
    hovermode='compare',
    xaxis1=XAxis(
        title='Trip Date',
        titlefont=Font(
            size=11.0,
            color='#262626'
        ),
        range=[X.min(),X.max()],
        domain=[0.0, 1.0],
        type='date',
        showgrid=True,
        zeroline=False,
        showline=True,
        nticks=8,
        ticks='inside',
        tickfont=Font(
            size=10.0
        ),
        mirror='ticks',
        anchor='y1',
        side='bottom'
    ),
    yaxis1=YAxis(
        title='Price',
        titlefont=Font(
            size=11.0,
            color='#262626'
        ),
        range=[ymin, ymax],
        domain=[0.0, 1.0],
        type='linear',
        showgrid=True,
        zeroline=False,
        showline=True,
        nticks=7,
        ticks='inside',
        tickfont=Font(
            size=10.0
        ),
        mirror='ticks',
        anchor='x1',
        side='left'
    )
)



#Short function for pushing private graphs to plotly
def private_plot(*args, **kwargs):
    kwargs['auto_open'] = True     #Controls whether a new tab is opened in your browser with the new plot
    url = py.plot(*args, **kwargs)
    return (url)



def main(): 
    fig = Figure(data=trace_list, layout=layout)
    url = private_plot(fig,  filename='%s/%s' %(folder_name, graph_title), world_readable=False)
    Range('Dashboard', 'B14').value = url