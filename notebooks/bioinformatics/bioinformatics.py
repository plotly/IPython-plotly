
# coding: utf-8

# # Visualizing biological data: exploratory bioinformatics with plot.ly

# ####About the author:
# Oxana is a data scientist based in Stockholm, Sweden. She is studying for a PhD in Bioinformatics, exploring molecular evolution patterns in eukaryotes. You can follow Oxana on Twitter [@Merenlin](http://twitter.com/Merenlin) or read [her blog](http://merenlin.com).
# 
# ###Introduction
# This notebook will give you the recipes of the most popular data visualizations I encounter in my work as a bioinformatician. If you always wondered what bioinformatics is all about or would like to create interactive
# visualization for your genomic data using [plot.ly](https://plot.ly/python/), this is the place to start. 
# 
# We will be working with real [gene expression](http://en.wikipedia.org/wiki/Gene_expression) data obtained by [Cap Analysis of Gene Expression(CAGE)](http://en.wikipedia.org/wiki/Cap_analysis_gene_expression) from human samples by the [FANTOM5](http://fantom.gsc.riken.jp/5/) consortium. We will be following a typical workflow of a bioinformatician exploring new data, looking for the outliers: interesting genes or samples, or general patterns in the data. In the end, you'll get the idea of the challenges and upsides of creating interactive visualizations of biological data using plot.ly Python API. 

# ### Obtaining the data

# FANTOM5 provides high precision data of thousands of human and mouse samples. The vastness of this data can be overwhelming and operating it locally is challenging. Luckily, there are many tools out there to make our life easier.  
# For creating a small data subset we can work with in this tutorial, I used [TET: Fantom 5 Table Extraction tool](http://fantom.gsc.riken.jp/5/tet). I picked a few human samples, mostly brain tissues with a few outliers, like uterus and downloaded a tab-separated file from the website. For more advanced data extraction, it's good to have a look at [TET's API](https://github.com/Hypercubed/TET/blob/master/README.md). 
# I have picked normalized tpm(tags per million) and annotated data, so we can focus only on processed data for protein coding genes. All data files for this notebook are available on figshare: http://dx.doi.org/10.6084/m9.figshare.1430029

# ###Loading the dataset

# We are loading the data from the .tsv file, skipping the first two columns (00Annotation and short_description).

# In[47]:

import numpy as np
import pandas as pd

data = np.genfromtxt("http://figshare.com/download/file/2087487/1",
                     comments="#", usecols=range(2,73,1), names=True, dtype=object, delimiter="\t")
df = pd.DataFrame(data)
print "Number of genes: " + str(len(df))
df.head()


#  Let's also make sure that we filter out those genes for which the [Uniprot](http://www.uniprot.org/) Id is unknown. That will reduce our data, besides, we are only interested in proteins in this analysis. 

# In[48]:

uniprot_clean = [x for x in df['uniprot_id'] if (x != 'NA') and ((x != ''))]
df=df[df["uniprot_id"].isin(uniprot_clean)]
print "Number of genes: " + str(len(df))
df.head()


# ###1. MA scatter plot comparing newborn and adult tissues

# [MA plot](http://en.wikipedia.org/wiki/MA_plot) is a popular visualization tool coming from the microarray analysis. It allows researchers to explore true statistical differences between the two samples, arrays or other observations. We are going to look at the two samples of substantia nigra tissues from the brain of an adult and a newborn person. How do their genetic profiles differ? 
# 
# Firs, let's subset our big dataframe to only include the samples of interest. We will also prefilter the data to not include genes that are not expressed in these tissues. 
# 

# In[49]:

df_MA = df[["uniprot_id",'substantia_nigra_adult_donor10258CNhs1422410371105G2','substantia_nigra_newborn_donor10223CNhs1407610358105E7']]
df_MA.columns = ['gene','adult', 'newborn']
df_MA[['adult','newborn']] = df_MA[['adult', 'newborn']].astype(float) 
df_MA = df_MA[(df_MA.T != 0).any()]  #remove rows with all zeros


# There are many different methods of computing the average expression level(A) between the two observations and
# the mean variation(M). To keep things simple, for this example we will just compare the sum on the x-axis vs the minus on the y-axis.
# Our data is already normalized and preprocessed, so this will be enough to find the clear outliers. 
# 
# Here lies the firs problem with web-based interactive visualizations. Plot.ly at the moment has a very hard time rendering more than 40k points in a scatter plot. So for the sake of this example, I'll plot a subset of the data. 

# In[50]:

import plotly.plotly as py
from plotly.graph_objs import *

A = df_MA['adult'] + df_MA['newborn']
M = df_MA['adult'] - df_MA['newborn']

trace = Scatter(
    x=A[1:1000],
    y=M[1:1000],
    mode='markers',
    name="substantia nigra",
    text=df_MA['gene'][1:1000],
    marker=Marker(
        size=5,
        line=Line(
            width=0.5),
        opacity=0.8))

layout = Layout(showlegend=True,
                title="MA plot of gene expression in adult and newborn samples of substantia nigra",
                xaxis=XAxis(
                    title='A',
                ),
               yaxis=YAxis(
                    title='M',
                ),
                )
fig = Figure(data=Data([trace]), layout=layout)
py.iplot(fig)


# Now we can already start exploring some of the genes, that behave differently in adult vs newborn samples. 

# ### 2. Histograms of expression breadth and average expression levels

# Another timeless visualization for exploratory data analysis is histogram. Here we don't need to subset our data anymore, for plots like these Plot.ly's capacity is up to 100k points. Let's see how expression breadth(in how many tissues the gene is expressed) and average expression levels look like for all of the samples. 
# 
# Just use the "domain" variable to regulate where the axis of each subplot are. 

# In[51]:

df['breadth'] =  (df[df.columns[1:].values.tolist()].astype('float')

>0).sum(axis=1)

df['avg'] = df[df.columns[1:].values.tolist()].astype('float').mean(axis=1)

trace1 = Histogram(
        name="expression breadth",
        x = df['breadth'],
        marker=Marker(
        line=Line(
            color='grey',
            width=0
        ),
        opacity=0.75
        ),
)

trace2 = Histogram(
        name="average expression",
        x = df['avg'],
        marker=Marker(
        line=Line(
            color='grey',
            width=0
        ),
        opacity=0.75
        ),
    xaxis='x2',
    yaxis='y2'
    )


layout = Layout(
    title="Exploring the distributions",
    xaxis=XAxis(
        title='breadth',
         domain=[0, 0.45]
    ),
    xaxis2=XAxis(
        title='average expression',
        domain=[0.55, 1],
    ), 
    yaxis2=YAxis(
        anchor='x2'
    )
)

fig = Figure(data=Data([trace1, trace2]), layout=layout)
py.iplot(fig)


# Here is where interactive visualization comes in handy. Average expression level distribution looks very wide because of a few outliers, - highly expressed genes and most of the genes actually being expressed at a very low level. But instead of trying to adjust the limits on the x-axis, we can just zoom in on the interesting area. Try it!

# ### 3. Scatter plot with a trend line

# This kind of plot must be the most popular way to visualize a trend in biological data. We seek clear
# and simple patterns demonstrating the relationships between different biological parameters or observations.
# Plot.ly's Python API does not come with out-of-the-box tools for plotting trend lines, but numpy has all we need. 
# 
# Let's say we want to plot the relationship between the breadth of expression and the average level. Again, for speed and simplicity, we only take the first 1000 genes in our data frame. Let's try to fit a polinomial function to our data points and plot both at the same time. By using plot.ly it's simple, just send the regression line trace to the same figure.  

# In[52]:

x = df['breadth'][1:1000]
y = df['avg'][1:1000]
coefficients = np.polyfit(x, y, 6)
polynomial = np.poly1d(coefficients)
r_x = np.arange(0, 72, 0.5)
r_y = polynomial(r_x)

trace1 = Scatter(
    x=x,
    y=y,
    mode='markers',
    name="expression levels",
    text=df['uniprot_id'][1:1000],
    marker=Marker(
        size=5,
        line=Line(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5),
        opacity=0.2))

trace2 = Scatter(
    mode='lines+markers',
    x=r_x, 
    y=r_y,
    marker=Marker(
        size=5,
        line=Line(
            color='purple',
            width=0.5),
        opacity=0.5),
    name="breadth regression")

layout = Layout(
    title="Breadth of expression vs average expression level",
    xaxis=XAxis(
        title='breadth',
    ),
    yaxis=YAxis(
        title='average expression',
    ),
)
fig = Figure(data=Data([trace1, trace2]), layout=layout)
py.iplot(fig)


# ### 4. Heatmap of gene expression

# Heatmap is another great way to visualize big amounts of data. It allows to clearly see the outliers and explore the 
# general clustering patterns. Are genes in different tissues, but the same donor expressed similarly or do the same tissues
# from different donors tend to cluster together? Do brains of newborns and adults differ in gene expression patterns? 
# Heatmaps of gene expression can give you good leads to questions like these. 

# There is one catch with generating a heatmap for biological samples using plot.ly. Labels of the heatmap will actually
# be coordinates on the x and y axis. For the plot to look less cluttered, I have removed the grid and set dtick to 1. Setting autotick to False also proved useful in order to see all the samples correctly labeled. 
# 
# To improve readability, one often also needs to process samples names. In our data, as you probably noticed, sample names include everything: tissue name, annotation, donor, age. The name becomes long and impossible to display in a plot. Simple shortening will not work with plot.ly though, since the coordinates must be unique! 
# 
# For this tutorial I've cheated a bit, by just adding an integer to each shortened name, I'm sure you can handle the string processing of your samples names on your own ;-)

# In[53]:

from scipy.spatial.distance import pdist, squareform

cols = [col for col in df.columns if col not in ['breadth', 'uniprot_id', 'avg']]
short_cols = [col[0:20] for col in cols]
short_cols = [short_cols[i] + str(i) for i in range(1,len(short_cols),1)]
data_dist = pdist(df[cols].as_matrix().transpose())

data = Data([
    Heatmap(
        z=squareform(data_dist), colorscale='YIGnBu',
        x=short_cols,
        y=short_cols,     # y-axis labels
    )
])

layout = Layout(
    title='Transcription profiling of human brain samples',
    autosize=False,
    margin=Margin(
        l=200,
        b=200,
        pad=4
    ),
    xaxis=XAxis(
        showgrid=False, # remove grid
        autotick=False, # custom ticks
        dtick=1,        # show 1 tick per day
    ),
    yaxis=YAxis(
        showgrid=False,   # remove grid
        autotick=False,   # custom ticks
        dtick=1           # show 1 tick per day
    ),
)
fig = Figure(data=data, layout=layout)
py.iplot(fig, width=900, height=900)


# ### 5. Network of gene interactions

# Now at some point in our biological investigations, we've got to dig deeper and look at concrete genes/proteins we found interesting. 
# 
# If you go back to our first plot, you'll see that one of the points that stand out corresponds to Q16352(Alpha-internexin, AINX_HUMAN). This gene demonstrates both high level of expression in substantia nigra and the difference between adult and newborn samples is also significant. Which kind of makes sense, since this protein is involved in the morphogenesis of neurons. One of the ways to find out more about a protein is to look at it's interaction networks. 
# 
# I've downloaded the interaction network in tab-separated format from a popular database [string-db.org](http://string-db.org/), so there is nothing novel in plotting it, we are merely reproducing the graph on their website, but, hopefully, you'll be able to use it for your future contributions to science! 

# In[54]:

import networkx as nx

import plotly.plotly as py
from plotly.graph_objs import *

x = np.genfromtxt('http://figshare.com/download/file/2088824', delimiter="\t", names=True, usecols=[0,1,14],
                  dtype=['S5','S5','f8'])
labels = x.dtype.names

G=nx.Graph()
G.add_weighted_edges_from(x)

pos=nx.spring_layout(G)

edge_trace = Scatter(x=[], y=[], mode='lines')
for edge in G.edges():
    x0, y0 = pos[edge[0]] 
    x1, y1 = pos[edge[1]] 
    edge_trace['x'] += [x0, x1, None]
    edge_trace['y'] += [y0, y1, None]

node_trace = Scatter(x=[], y=[], mode='markers+text',
                     text=G.nodes(),
                     textposition='top',
                     marker=Marker(size=10))
for node in G.nodes():
    x, y = pos[node]
    node_trace['x'].append(x)
    node_trace['y'].append(y)
   
fig = Figure(data=Data([edge_trace, node_trace]),
             layout=Layout(title='AINX_HUMAN interaction network',
                           showlegend=False, xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False)))

py.iplot(fig)



# Now you see our protein under it's gene name(INA) in the center of the graph. 

# Now, that's all, folks! I hope you enjoyed this intro to exploratory bioinformatics and got inspired to create beautiful interactive visualizations for your biological data. 
